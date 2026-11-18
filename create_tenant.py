import sys
from app.db import init_db, create_tenant

init_db()
print("API key (shown once):", create_tenant(sys.argv[1], sys.argv[2]))