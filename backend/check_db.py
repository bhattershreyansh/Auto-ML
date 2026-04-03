import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add the current directory to sys.path so we can import database
sys.path.append(os.getcwd())

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ DATABASE_URL not found in environment.")
    sys.exit(1)

print(f"🔍 Testing connection to: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")

try:
    # Only use SQLite-specific connect_args if the URL starts with sqlite
    is_sqlite = DATABASE_URL.startswith("sqlite")
    engine_args = {"connect_args": {"check_same_thread": False}} if is_sqlite else {}
    
    engine = create_engine(DATABASE_URL, **engine_args)
    
    with engine.connect() as conn:
        # Check if tables exist
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
        tables = [row[0] for row in result]
        
        if tables:
            print(f"✅ Connection successful! Found tables: {', '.join(tables)}")
        else:
            print("✅ Connection successful, but no tables found in 'public' schema yet.")
            
except Exception as e:
    print(f"❌ Connection failed: {e}")
