import re
from collections.abc import Mapping
from typing import Any, Dict, Union

#: Placeholder names may address a section with ``/``, as in
#: ``{outputs/draft}`` or ``{outputs/news/draft}``. A bare ``{draft}`` is
#: still resolved by scope order, so existing templates are unaffected.
PLACEHOLDER_PATTERN = r"\{([\w/]+)\}"

PATH_SEPARATOR = "/"


def resolve_template_path(fields: Mapping, path: str) -> "tuple[bool, Any]":
    """Resolve a possibly-sectioned ``path`` against a mapping.

    Returns ``(found, value)``. A mapping that knows its own sections — such
    as ``PipelineResults`` — resolves the path itself; anything else is walked
    as nested mappings, so a plain dict of dicts works too.
    """

    resolver = getattr(fields, "resolve_path", None)
    if callable(resolver):
        return resolver(path)

    if PATH_SEPARATOR not in path:
        return (path in fields), fields.get(path)

    current = fields
    for part in path.split(PATH_SEPARATOR):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current


class MessageTemplate:
    """A prompt template with ``{placeholder}`` fields.

    Build one with :meth:`from_string` or :meth:`from_file`, then fill it with
    :meth:`generate_message_content`. A placeholder may address a nested value
    with ``/``, as in ``{outputs/draft}``.

    Attributes:
        template (str): The template string containing placeholders.
    """

    def __init__(self, template_file=None, template_string=None):
        """
        Initialize a PromptTemplate instance.

        Args:
            template_file (str): The path to a file containing the template.
            template_string (str): The template string.
        """
        if template_file:
            with open(template_file, "r") as file:
                self.template = file.read()
        elif template_string:
            self.template = template_string
        else:
            raise ValueError(
                "Either 'template_file' or 'template_string' must be provided"
            )

    @classmethod
    def from_string(cls, template_string) -> "MessageTemplate":
        """
        Create a PromptTemplate instance from a string.

        Args:
            template_string (str): The template string.

        Returns:
            PromptTemplate: Created PromptTemplate instance.
        """
        return cls(template_string=template_string)

    @classmethod
    def from_file(cls, template_file) -> "MessageTemplate":
        """
        Create a PromptTemplate instance from a file.

        Args:
            template_file (str): The path to a file containing the template.

        Returns:
            PromptTemplate: Created PromptTemplate instance.
        """
        with open(template_file, "r") as file:
            template_string = file.read()
        return cls(template_string=template_string)

    @staticmethod
    def _remove_empty_placeholders(text):
        """
        Remove lines that contain only the empty placeholder.

        Args:
            text (str): The text containing placeholders.

        Returns:
            str: Text with empty placeholders removed.
        """
        lines = text.split("\n")
        processed_lines = []
        for line in lines:
            if "__EMPTY_TEMPLATE_FIELD__" in line:
                new_line = line.replace("__EMPTY_TEMPLATE_FIELD__", "")
                if new_line.strip():
                    processed_lines.append(new_line)
            else:
                processed_lines.append(line)
        return "\n".join(processed_lines)

    def generate_message_content(
        self,
        template_fields: Dict[str, Any] = None,
        remove_empty_template_field: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Generate a prompt by replacing placeholders in the template with values.

        Args:
            template_fields (Dict[str, Any], optional): The template fields as a dictionary.
            remove_empty_template_field (bool): If True, removes lines with empty placeholders.
            **kwargs: Additional keyword arguments to be used as template fields.

        Returns:
            str: The generated prompt.
        """
        # A single mapping is used directly rather than being flattened into
        # kwargs, so a sectioned mapping keeps its structure and paths such as
        # {outputs/draft} can be resolved.
        if template_fields is not None and not kwargs:
            all_fields = template_fields
        elif template_fields is None and kwargs:
            all_fields = kwargs
        else:
            all_fields = {**(template_fields or {}), **kwargs}

        def lookup(placeholder):
            """Return ``(found, text)`` for a placeholder name or path."""

            found, value = resolve_template_path(all_fields, placeholder)
            if not found or value is None:
                return False, None
            return True, value if isinstance(value, str) else str(value)

        if not remove_empty_template_field:

            def replace_placeholder(match):
                found, text = lookup(match.group(1))
                return text if found else match.group(0)

            return re.sub(PLACEHOLDER_PATTERN, replace_placeholder, self.template)

        def replace_placeholder(match):
            found, text = lookup(match.group(1))
            if found:
                return text
            if PATH_SEPARATOR in match.group(1):
                # Widening the pattern to accept "/" means text that was always
                # literal — "{a/b}" in a prompt — would suddenly be treated as a
                # placeholder and blanked. An unresolved *path* is left verbatim,
                # which keeps such text intact and makes a mistyped path visible
                # instead of silently empty. Bare names keep the old behaviour.
                return match.group(0)
            return "__EMPTY_TEMPLATE_FIELD__"

        # Initial placeholder replacement
        prompt = re.sub(PLACEHOLDER_PATTERN, replace_placeholder, self.template)

        return self._remove_empty_placeholders(prompt)

class ChatFormatter:
    def __init__(self, template, role_names: Dict[str, str] = None):
        self.template = template
        self.role_names = role_names or {}

    def format_messages(self, messages):
        formatted_chat = []
        for message in messages:
            msg_dict = message.to_dict()
            role = msg_dict["role"]
            content = msg_dict['content']
            display_name = self.role_names.get(role, role.capitalize())
            formatted_message = self.template.format(role=display_name, content=content)
            formatted_chat.append(formatted_message)
        return '\n'.join(formatted_chat)