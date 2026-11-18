import json
import sys

from app.db import update_tenant_config

tenant, key, raw = sys.argv[1:4]
try:
    value = json.loads(raw)
except json.JSONDecodeError:
    value = raw
update_tenant_config(tenant, {key: value})
print("updated", tenant, key, "=", value)