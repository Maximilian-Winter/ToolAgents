from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

collect_ignore = [
    'advanced_chat_agent_test.py',
    'example_rag_bars.py',
    'test_advanced_agent.py',
    'test_advanced_chat_formatter.py',
    'test_chat_api_agent.py',
    'test_llama_31_agent.py',
    'test_mistral_agent.py',
    'test_mistral_agent_memory.py',
    'test_ollama_agent.py',
]
