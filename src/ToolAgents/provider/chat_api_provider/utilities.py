from typing import List


def clean_history_messages(history_messages: List[dict]) -> List[dict]:
    clean_messages = []
    for msg in history_messages:
        if "id" in msg:
            msg.pop("id")
        clean_messages.append(msg)

    return clean_messages
