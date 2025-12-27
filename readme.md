# RAG Server Application

A sophisticated locally-hosted Retrieval-Augmented Generation (RAG) server built with FastAPI, featuring advanced document processing, hybrid search capabilities, and intelligent chat functionalities.

## 🚀 Features

- **Document Processing**: Support for multiple file formats (PDF, DOCX, HTML, PPTX, TXT, MD)
- **Hybrid Search**: Combines vector similarity and full-text search with configurable weights
- **Multi-Query RAG**: Generates multiple search queries for comprehensive retrieval
- **Rerank Fusion**: Advanced reranking for optimal search results
- **Real-time Chat**: Interactive chat interface with document context
- **Background Processing**: Celery-powered asynchronous document processing
- **Vector Storage**: PostgreSQL with pgvector for efficient similarity search
- **Authentication**: Clerk-based user authentication and authorization
- **Observability**: OpenTelemetry integration for LLM cost tracking and performance monitoring (planned)

## 🛠️ Tech Stack

### Core Technologies
- **FastAPI** - High-performance Python web framework
- **PostgreSQL** - Primary database with vector search capabilities
- **pgvector** - Vector similarity search extension
- **Supabase** - Backend-as-a-Service platform
- **Redis** - Task queue and caching
- **Celery** - Distributed task queue for background processing

### AI/ML Stack
- **LangChain** - Framework for building LLM applications
- **Google Gemini** - Large language model for chat and summarization
- **Ollama** - Local LLM hosting (alternative), Local Embeddings and language models
- **Unstructured** - Document parsing and chunking

### Infrastructure
- **Docker** - Containerization
- **Supabase S3 Storage** - File storage and management
- **Clerk** - Authentication and user management
- **OpenTelemetry** - Observability and LLM cost tracking (planned)

## 📋 Prerequisites

- Python 3.12+
- PostgreSQL with pgvector extension
- Redis server
- Docker (required - will be implemented next)
- Supabase project with Storage enabled
- Clerk application

## ⚙️ Installation

### 1. Clone and Setup Environment

```bash
cd server
pip install -r requirements.txt  # or use poetry install
```

### 2. Environment Configuration

Create a `.env` file with the following variables:

```env
# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379/0

# Supabase Storage
SUPABASE_STORAGE_BUCKET=your_storage_bucket_name

# Authentication
CLERK_SECRET_KEY=your_clerk_secret_key
CLERK_JWT_VERIFICATION_KEY=your_clerk_jwt_key

# AI Models
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key

# Observability (planned)
OTEL_EXPORTER_OTLP_ENDPOINT=your_otel_endpoint
OTEL_SERVICE_NAME=rag-server
OTEL_RESOURCE_ATTRIBUTES=service.name=rag-server,service.version=1.0.0

# Application
ENVIRONMENT=development
```

### 3. Database Setup

Run the Supabase migrations:

```bash
# Apply database migrations
supabase db push
```

### 4. Start Services

#### Using Docker (Recommended)

```bash
# Start PostgreSQL with pgvector and Redis
docker-compose up -d postgres redis

# Start Celery worker
celery -A tasks worker --loglevel=info --pool=threads

# Start FastAPI server
python main.py
```

#### Manual Setup

```bash
# Start Redis (WSL)
sudo systemctl start redis-server.service

# Start Celery worker
celery -A tasks worker --loglevel=info --pool=threads

# Start development server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🏗️ Architecture

### Core Components

1. **FastAPI Application** (`main.py`)
   - RESTful API endpoints
   - CORS middleware
   - Route organization by feature

2. **Authentication** (`src/auth.py`)
   - Clerk JWT verification
   - User authorization middleware

3. **Routers** (`src/routers/`)
   - `users.py` - User management
   - `projects.py` - Project CRUD operations
   - `chats.py` - Chat and messaging
   - `files.py` - Document upload and management

4. **Services** (`src/services/`)
   - `supabase.py` - Database and storage client
   - `s3.py` - Supabase Storage operations
   - `llm.py` - Language model integrations

5. **Background Tasks** (`tasks.py`)
   - Document processing pipeline
   - Embedding generation
   - Chunk storage

### Document Processing Pipeline

1. **Upload** → Document uploaded to Supabase Storage
2. **Partition** → Extract text and structure using Unstructured
3. **Chunk** → Split into semantic chunks
4. **Summarize** → Generate summaries for complex elements
5. **Vectorize** → Create embeddings using OpenAI/Google models
6. **Store** → Save to PostgreSQL with pgvector

### Search Architecture

The application implements a hybrid search approach:

1. **Vector Search** - Semantic similarity using embeddings
2. **Full-Text Search** - PostgreSQL FTS for keyword matching
3. **Hybrid Fusion** - Weighted combination of both approaches
4. **Multi-Query** - Generate multiple search perspectives
5. **Reranking** - Final result optimization

## 📡 API Endpoints

### Projects
- `GET /api/projects` - List user projects
- `POST /api/projects` - Create new project
- `GET /api/projects/{project_id}` - Get project details
- `DELETE /api/projects/{project_id}` - Delete project

### Documents
- `POST /api/files/upload` - Upload document
- `GET /api/projects/{project_id}/documents` - List project documents
- `DELETE /api/files/{document_id}` - Delete document

### Chats
- `POST /api/chats` - Create new chat
- `GET /api/projects/{project_id}/chats` - List project chats
- `POST /api/chats/{chat_id}/messages` - Send message
- `GET /api/chats/{chat_id}/messages` - Get chat history

### Health
- `GET /health` - Service health check
- `GET /` - Root endpoint

## 🔧 Configuration

### RAG Settings

Each project can be configured with:

- **Embedding Model**: OpenAI, Google, or local models
- **RAG Strategy**: Single-query, multi-query, or hybrid
- **Search Parameters**: Chunk count, similarity threshold
- **Reranking**: Enable/disable and model selection
- **Hybrid Weights**: Vector vs. keyword search balance

### Supported File Types

- PDF documents
- Microsoft Word (DOCX)
- PowerPoint (PPTX)
- HTML files
- Plain text (TXT)
- Markdown (MD)

## 🚀 Deployment

### Docker Deployment

```bash
# Build the application
docker build -t rag-server .

# Run with Docker Compose
docker-compose up -d
```

### Local Hosting Considerations

1. **Database**: Local PostgreSQL with pgvector extension
2. **Redis**: Local Redis server or Docker container
3. **Storage**: Supabase Storage with local configuration
4. **Scaling**: Multiple Celery workers for document processing
5. **Monitoring**: Local logging and health checks
6. **Security**: Secure environment variable management
7. **Backup**: Regular database and storage backups

## 🔍 Monitoring and Debugging

### Celery Monitoring

```bash
# Monitor Celery tasks
celery -A tasks inspect active

# View worker status
celery -A tasks inspect stats
```

### Database Queries

The application uses several optimized queries:
- Vector similarity search with HNSW indexes
- Full-text search with GIN indexes
- Hybrid search combining multiple strategies

### Logging

Application logs include:
- Request/response cycles
- Document processing stages
- Search query performance
- Error tracking and debugging

## 📝 Development

### Adding New File Types

1. Add partitioning logic in `tasks.py`
2. Update file type validation
3. Test processing pipeline

### Extending Search Capabilities

1. Modify search functions in `chats.py`
2. Update reranking algorithms
3. Add new similarity metrics

### Custom LLM Integration

1. Extend `llm.py` service
2. Add model-specific handling
3. Update configuration options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review logs for error details
3. Open an issue on GitHub
4. Contact the development team

---

Built with ❤️ using FastAPI, PostgreSQL, and modern AI technologies.