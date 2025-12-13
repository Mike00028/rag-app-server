from supabase import Client, create_client
from src.config.index import appConfig

supabase_url = appConfig["supabase_api_url"]
supabase_key = appConfig["supabase_secret_key"]

if not supabase_url or not supabase_key:
    raise ValueError("Supabase URL and secret key must be provided")

supabase: Client = create_client(supabase_url, supabase_key)