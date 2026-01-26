from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

from src.routers import chats
from src.routers import projects
from src.routers import users
from src.routers import files


app = FastAPI(
    title="My FastAPI Application",
    description="This is a sample FastAPI application with CORS enabled.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# users router
app.include_router(users.router)

# projects router
app.include_router(projects.router)

# chats router
app.include_router(chats.router)

# files router

app.include_router(files.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}


  

if __name__ == "__main__":
    # Allow environment to be set via command line: python main.py prod or python main.py production
    if len(sys.argv) > 1:
        env_arg = sys.argv[1].lower()
        if env_arg in ["local", "prod", "production"]:
            # Normalize to "local" or "prod"
            env = "prod" if env_arg in ["prod", "production"] else "local"
            os.environ["ENVIRONMENT"] = env
            print(f"🌍 Environment set to: {env}")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)