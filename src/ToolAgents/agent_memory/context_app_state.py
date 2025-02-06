import enum
import os
import yaml
import json
import xml.etree.ElementTree as ET
import re
from typing import Dict, Any
from ToolAgents import FunctionTool


class AppStateOperation(enum.Enum):
    create = "create"
    replace = "replace"
    append = "append"
    delete = "delete"

class ContextAppState:
    def __init__(self, initial_state_file: str = None):
        self.template_fields = {}
        if initial_state_file is not None:
            self.template_fields = self.load_yaml_initial_app_state(initial_state_file)

        def edit_app_state( state_name: str, operation: AppStateOperation, field_name: str, content: str = None):
            """
            Operation to perform on the app state. Can be 'create', 'replace', 'append' or 'delete'.
            Args:
                state_name(str): Name of app state to edit.
                operation(AppStateOperation): Operation to perform on the app state. Can be 'create', 'replace', 'append' or 'delete'.
                field_name(str): Name of field on app state to edit.
                content(str): Optional content(required for 'create', 'replace' and 'append' operations).
            """
            if operation == AppStateOperation.create:
                if content is None:
                    raise ValueError("Content for 'create' operation requires 'content' argument.")
                self.template_fields[state_name] = content
            elif operation == AppStateOperation.replace:
                if content is None:
                    raise ValueError("Content for 'replace' operation requires 'content' argument.")
                self.template_fields[state_name] = content
            elif operation == AppStateOperation.append:
                if content is None:
                    raise ValueError("Content for 'append' operation requires 'content' argument.")
                self.template_fields[state_name] += "\n" + content
            elif operation == AppStateOperation.delete:
                del self.template_fields[state_name]
            return "App state edited successfully."
        self.edit_app_state_tool = FunctionTool(edit_app_state)

    def load_yaml_initial_app_state(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            return {}
        try:
            with open(file_path, 'r') as file:
                yaml_content = yaml.safe_load(file)
                return self._process_yaml_content(yaml_content)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return {}

    def _process_yaml_content(self, content: Dict[str, Any]) -> Dict[str, str]:
        return {k: self._process_value(v) for k, v in content.items()}

    def _process_value(self, v, indent=0):
        if isinstance(v, list):
            if any(isinstance(item, dict) and 'name' in item for item in v):
                return '\n'.join(self._process_special_item(item, indent) for item in v)
            return '\n'.join(self._process_list_item(item, indent) for item in v)
        elif isinstance(v, dict):
            return self._process_dict(v, indent)
        elif isinstance(v, str):
            return v.strip()
        else:
            return str(v)

    def _process_list_item(self, item, indent=0):
        if isinstance(item, dict):
            result = []
            for key, value in item.items():
                if isinstance(value, list):
                    result.append(f"{' ' * indent}- {key}:")
                    for subitem in value:
                        result.append(f"{' ' * (indent + 2)}- {subitem}")
                else:
                    result.append(f"{' ' * indent}- {key}: {self._process_value(value, indent + 2)}")
            return '\n'.join(result)
        else:
            return f"{' ' * indent}- {item}"

    def _process_dict(self, d, indent=0):
        result = []
        for key, value in d.items():
            if isinstance(value, dict):
                result.append(f"{' ' * indent}{key}:")
                result.append(self._process_dict(value, indent + 2))
            elif isinstance(value, list):
                result.append(f"{' ' * indent}{key}:")
                result.append(self._process_value(value, indent + 2))
            else:
                result.append(f"{' ' * indent}{key}: {self._process_value(value, indent + 2)}")
        return '\n'.join(result)

    def _process_special_item(self, item, indent=0):
        result = []
        for key, value in item.items():
            if isinstance(value, list):
                result.append(f"{' ' * indent}{key}:")
                for subitem in value:
                    result.append(f"{' ' * (indent + 2)}- {subitem}")
            else:
                result.append(f"{' ' * indent}{key}: {value}")
        return '\n'.join(result)

    def update_from_xml(self, xml_string: str) -> None:
        xml_string = f"<root>{xml_string}</root>"
        try:
            root = ET.fromstring(xml_string)
            self._process_element(root)
        except ET.ParseError:
            self._update_from_regex(xml_string)

    def _process_element(self, element: ET.Element) -> None:
        for child in element:
            if len(child) == 0:  # If the element has no children
                key = child.tag.split('.')[-1]
                self.template_fields[key] = child.text.strip() if child.text else ""
            else:
                self._process_element(child)

    def _update_from_regex(self, content: str) -> None:
        sections = re.findall(r'<([\w.]+)>(.*?)</\1>', content, re.DOTALL)
        for section, content in sections:
            key = section.split('.')[-1]
            self.template_fields[key] = content.strip()

    def save_json(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self.template_fields, f, indent=2)

    def load_json(self, filename: str) -> None:
        with open(filename, "r") as f:
            self.template_fields = json.load(f)

    def get_field(self, key: str, default: Any = None) -> Any:
        return self.template_fields.get(key, default)

    def set_field(self, key: str, value: Any) -> None:
        self.template_fields[key] = value

    def get_fields(self) -> Dict[str, str]:
        return self.template_fields

    def set_fields(self, fields: dict[str, str]) -> None:
        self.template_fields.update(fields)

    def get_app_state_string(self, begin_section_marker: str = "<{section_name}>\n", end_section_marker: str = "\n</{section_name}>") -> str:
        output = ""
        for key, value in self.template_fields.items():
            output += begin_section_marker.format(section_name=key)
            output += value
            output += end_section_marker.format(section_name=key)
            output += "\n\n"
        return output

    def get_edit_tool(self):
        return self.edit_app_state_tool

    def __str__(self) -> str:
        return f"ContextAppState(fields: {len(self.template_fields)})"

