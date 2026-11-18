import sqlite3

c = sqlite3.connect("gateway.db")
rows = c.execute(
    """SELECT tenant_id,
              SUM(status='HIT'), SUM(status='MISS'), SUM(status='BYPASS'),
              ROUND(AVG(CASE WHEN status='HIT' THEN latency_ms END)),
              ROUND(AVG(CASE WHEN status='MISS' THEN latency_ms END))
       FROM requests GROUP BY tenant_id"""
).fetchall()
for tid, hits, misses, bypass, hit_ms, miss_ms in rows:
    cacheable = (hits or 0) + (misses or 0)
    rate = f"{100 * hits / cacheable:.0f}%" if cacheable else "n/a"
    print(f"{tid}: hit rate {rate} | hits {hits} misses {misses} bypass {bypass} "
          f"| avg hit {hit_ms} ms, avg miss {miss_ms} ms")