#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class ClaudeToToolAgentsConverter:
    def __init__(self, input_path: str, output_dir: Optional[str] = None):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir) if output_dir else self.input_path.parent / "toolagents_exports"
        self.output_dir.mkdir(exist_ok=True)

    def convert_sender_to_role(self, sender: str) -> str:
        role_mapping = {
            'human': 'user',
            'assistant': 'assistant'
        }
        return role_mapping.get(sender.lower(), 'assistant')

    def extract_message_content(self, message: Dict[str, Any]) -> Any:
        content_items = message.get('content', [])

        if not content_items:
            return message.get('text', '')

        if len(content_items) == 1:
            item = content_items[0]
            if item.get('type') == 'text':
                return item.get('text', '')

        content_list = []
        for item in content_items:
            if item.get('type') == 'text':
                content_list.append({
                    'type': 'text',
                    'text': item.get('text', '')
                })
            elif item.get('type') == 'thinking':
                content_list.append({
                    'type': 'text',
                    'text': f"[Thinking]\n{item.get('thinking', '')}"
                })
            elif item.get('type') == 'tool_use':
                tool_call = {
                    'type': 'tool_use',
                    'name': item.get('name', 'unknown_tool'),
                    'input': item.get('input', {})
                }
                content_list.append(tool_call)

        return content_list if len(content_list) > 1 else (content_list[0] if content_list else '')

    def detect_tool_calls(self, message: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        content_items = message.get('content', [])
        tool_calls = []

        for item in content_items:
            if item.get('type') == 'tool_use':
                tool_call = {
                    'id': f"call_{message.get('uuid', 'unknown')[:8]}",
                    'type': 'function',
                    'function': {
                        'name': item.get('name', 'unknown_tool'),
                        'arguments': json.dumps(item.get('input', {}))
                    }
                }
                tool_calls.append(tool_call)

        return tool_calls if tool_calls else None

    def convert_claude_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        sender = message.get('sender', 'unknown')
        role = self.convert_sender_to_role(sender)

        toolagents_message = {
            'role': role,
            'content': self.extract_message_content(message)
        }

        tool_calls = self.detect_tool_calls(message)
        if tool_calls:
            toolagents_message['tool_calls'] = tool_calls

        if message.get('created_at'):
            toolagents_message['timestamp'] = message['created_at']

        return toolagents_message

    def convert_conversation(self, conversation: Dict[str, Any], index: int) -> Dict[str, Any]:
        messages = []

        chat_messages = conversation.get('chat_messages', [])

        for msg in chat_messages:
            converted_msg = self.convert_claude_message(msg)
            messages.append(converted_msg)

        toolagents_format = {
            'conversation_id': conversation.get('uuid', f'conv_{index}'),
            'title': conversation.get('name', f'Conversation {index}'),
            'created_at': conversation.get('created_at', ''),
            'updated_at': conversation.get('updated_at', ''),
            'messages': messages,
            'metadata': {
                'source': 'claude_export',
                'original_uuid': conversation.get('uuid', ''),
                'message_count': len(messages)
            }
        }

        return toolagents_format

    def sanitize_filename(self, name: str) -> str:
        import re
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        name = name.strip()
        return name[:200] if len(name) > 200 else name

    def save_conversation(self, conversation_data: Dict[str, Any], index: int) -> Path:
        title = conversation_data.get('title', f'conversation_{index}')
        safe_title = self.sanitize_filename(title)

        timestamp = conversation_data.get('created_at', '')[:10]
        if timestamp:
            filename = f"{timestamp}_{safe_title}.json"
        else:
            filename = f"{safe_title}.json"

        output_path = self.output_dir / filename

        counter = 1
        while output_path.exists():
            if timestamp:
                filename = f"{timestamp}_{safe_title}_{counter}.json"
            else:
                filename = f"{safe_title}_{counter}.json"
            output_path = self.output_dir / filename
            counter += 1

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)

        return output_path

    def convert_all(self) -> bool:
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return False
        except FileNotFoundError:
            print(f"File not found: {self.input_path}")
            return False

        conversations = data.get('conversations', [])

        if not conversations:
            if isinstance(data, list):
                conversations = data
            else:
                conversations = [data]

        print(f"Found {len(conversations)} conversation(s) to convert")
        print(f"Output directory: {self.output_dir}\n")

        all_conversations = []

        for i, conversation in enumerate(conversations, 1):
            converted = self.convert_conversation(conversation, i)
            output_path = self.save_conversation(converted, i)

            all_conversations.append({
                'file': output_path.name,
                'title': converted['title'],
                'messages': len(converted['messages']),
                'created': converted.get('created_at', 'unknown')
            })

            print(f"  [{i}/{len(conversations)}] Converted: {output_path.name}")
            print(f"       Title: {converted['title']}")
            print(f"       Messages: {len(converted['messages'])}")

        combined_file = self.output_dir / "all_conversations.json"
        all_messages = []
        for conv in conversations:
            converted = self.convert_conversation(conv, conversations.index(conv) + 1)
            all_messages.extend(converted['messages'])

        combined_data = {
            'total_conversations': len(conversations),
            'total_messages': len(all_messages),
            'export_date': datetime.now().isoformat(),
            'source': 'claude_export',
            'messages': all_messages
        }

        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)

        summary_file = self.output_dir / "conversion_summary.json"
        summary = {
            'conversion_date': datetime.now().isoformat(),
            'total_conversations': len(conversations),
            'output_directory': str(self.output_dir),
            'conversations': all_conversations
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Successfully converted {len(conversations)} conversation(s)")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Summary file: {summary_file}")
        print(f"🔗 Combined file: {combined_file}")

        return True


def main():
    if len(sys.argv) < 2:
        print("Claude to ToolAgents Converter")
        print("==============================")
        print("\nUsage: python claude_to_toolagents.py <path_to_conversations.json> [output_directory]")
        print("\nExample:")
        print("  python claude_to_toolagents.py ~/Downloads/conversations.json")
        print("  python claude_to_toolagents.py conversations.json ./toolagents_data")
        print("\nThis will convert Claude's export format to ToolAgents chat history format.")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    converter = ClaudeToToolAgentsConverter(input_path, output_dir)
    success = converter.convert_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()