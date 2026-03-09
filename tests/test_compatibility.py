from pathlib import Path
import shutil
import uuid

from ToolAgents.data_models.chat_history import ChatHistory
from ToolAgents.messages import ChatHistory as CompatChatHistory, ChatMessage
from ToolAgents.provider import AnthropicSettings, OpenAISettings
from ToolAgents.provider.completion_provider.default_implementations import LlamaCppProviderSettings
from ToolAgents.utilities import RecursiveCharacterTextSplitter, SimpleTextSplitter


def _compat_tmp_file() -> Path:
    root = Path('tests') / '.compat_tmp'
    root.mkdir(exist_ok=True)
    return root / f'{uuid.uuid4()}.json'


def test_compatibility_imports_are_available():
    assert CompatChatHistory is ChatHistory
    assert ChatMessage is not None
    assert SimpleTextSplitter is not None
    assert RecursiveCharacterTextSplitter is not None


def test_legacy_chat_history_format_loads():
    history_file = _compat_tmp_file()
    try:
        history_file.write_text(
            '[{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello"}]',
            encoding='utf-8',
        )
        history = ChatHistory.load_from_json(str(history_file))
    finally:
        if history_file.exists():
            history_file.unlink()
        if history_file.parent.exists() and not any(history_file.parent.iterdir()):
            history_file.parent.rmdir()

    assert history.get_message_count() == 2
    assert history.to_list()[0].get_as_text() == 'You are helpful.'
    assert history.to_list()[1].get_as_text() == 'Hello'


def test_legacy_chat_history_instance_helpers():
    history_file = _compat_tmp_file()
    try:
        history_file.write_text('[{"role": "user", "content": "Ping"}]', encoding='utf-8')
        history = ChatHistory()
        history.load_history(str(history_file))
        history.add_list_of_dicts([{'role': 'assistant', 'content': 'Pong'}])
    finally:
        if history_file.exists():
            history_file.unlink()
        if history_file.parent.exists() and not any(history_file.parent.iterdir()):
            history_file.parent.rmdir()

    assert history.get_message_count() == 2
    assert history.to_list()[-1].get_as_text() == 'Pong'


def test_legacy_provider_settings_aliases_work():
    openai_settings = OpenAISettings()
    anthropic_settings = AnthropicSettings()

    openai_settings.temperature = 0.2
    anthropic_settings.top_p = 0.8

    assert openai_settings.temperature == 0.2
    assert anthropic_settings.top_p == 0.8


def test_llama_cpp_settings_are_instantiable_and_flattened():
    settings = LlamaCppProviderSettings()
    settings.set_max_new_tokens(1024)
    settings.set_stop_tokens(['assistant'])
    settings.set_extra_body({'mirostat': 2})

    payload = settings.to_dict()

    assert payload['n_predict'] == 1024
    assert payload['stop'] == ['assistant']
    assert payload['mirostat'] == 2
