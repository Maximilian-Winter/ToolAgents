import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

from ToolAgents.data_models.chat_history import ChatHistory
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.knowledge.text_processing.text_splitter import (
    RecursiveCharacterTextSplitter,
    SimpleTextSplitter,
)
from ToolAgents.provider.completion_provider.default_implementations import (
    LlamaCppProviderSettings,
)


TEST_ROOT = Path(__file__).resolve().parent


def test_public_api_imports_are_available():
    assert ChatHistory is not None
    assert ChatMessage is not None
    assert SimpleTextSplitter is not None
    assert RecursiveCharacterTextSplitter is not None


def test_chat_history_json_roundtrip():
    history = ChatHistory()
    history.add_system_message('You are helpful.')
    history.add_user_message('Hello')

    history_path = TEST_ROOT / 'chat_history_roundtrip.json'
    history.save_to_json(str(history_path))
    loaded = ChatHistory.load_from_json(str(history_path))
    history_path.unlink(missing_ok=True)

    assert loaded.get_message_count() == 2
    assert loaded.get_messages()[0].get_as_text() == 'You are helpful.'
    assert loaded.get_messages()[1].get_as_text() == 'Hello'



def test_llama_cpp_settings_are_instantiable_and_flattened():
    settings = LlamaCppProviderSettings()
    settings.set_max_new_tokens(1024)
    settings.set_stop_tokens(['assistant'])
    settings.set_extra_body({'mirostat': 2})

    payload = settings.to_dict()

    assert payload['n_predict'] == 1024
    assert payload['stop'] == ['assistant']
    assert payload['mirostat'] == 2


def test_completion_provider_import_does_not_require_mistral_client():
    code = textwrap.dedent(
        """
        import types
        import sys

        sys.modules['mistralai'] = types.ModuleType('mistralai')

        from ToolAgents.provider.completion_provider.default_implementations import (
            LlamaCppProviderSettings,
        )

        assert LlamaCppProviderSettings is not None
        assert 'ToolAgents.provider.chat_api_provider.mistral' not in sys.modules
        """
    )

    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=TEST_ROOT.parent,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_removed_legacy_modules_are_not_public_api():
    import ToolAgents.agent_memory as agent_memory
    import ToolAgents.agents as agents

    assert "AdvancedAgent" not in agents.__all__
    assert "AgentConfig" not in agents.__all__
    assert "SemanticMemory" not in agent_memory.__all__
    assert "ContextAppState" not in agent_memory.__all__

    for module_name in (
        "ToolAgents.agents.advanced_agent",
        "ToolAgents.agent_memory.semantic_memory",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{module_name} should not be importable")
