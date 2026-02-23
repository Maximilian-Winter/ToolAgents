from execution_integrity_core import ExecutionIntegrityCore

ei = ExecutionIntegrityCore()

ei.record("tool_call", {"tool": "search", "q": "agent execution"}, {"ok": True}, ts=1700000000.0)
ei.record("tool_call", {"tool": "calc", "expr": "2+2"}, {"result": 4}, ts=1700000001.0)

print("Verification before tamper:", ei.verify())

ei.export(filename="execution_log.json", exported_at=1700000002.0)

# 篡改演示
ei.chain[0]["output"] = {"ok": False}
print("Verification after tamper:", ei.verify())

print("Exported: execution_log.json")
print("Run verifier: python3 verify_export.py execution_log.json")
