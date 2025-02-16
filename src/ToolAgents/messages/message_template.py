import re
from typing import Union, Dict, Any


class MessageTemplate:
    """
    Class representing a prompt template.

    Methods:
        generate_prompt(*args, **kwargs) -> str:
        Generate a prompt by replacing placeholders in the template with values.

    Class Methods:
        from_string(template_string: str) -> PromptTemplate:
        Create a PromptTemplate from a string.
        from_file(template_file: str) -> PromptTemplate:
        Create a PromptTemplate from a file.

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
    def from_string(cls, template_string) -> 'MessageTemplate':
        """
        Create a PromptTemplate instance from a string.

        Args:
            template_string (str): The template string.

        Returns:
            PromptTemplate: Created PromptTemplate instance.
        """
        return cls(template_string=template_string)

    @classmethod
    def from_file(cls, template_file) -> 'MessageTemplate':
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
        lines = text.split('\n')
        processed_lines = []
        for line in lines:
            if '__EMPTY_TEMPLATE_FIELD__' in line:
                new_line = line.replace('__EMPTY_TEMPLATE_FIELD__', '')
                if new_line.strip():
                    processed_lines.append(new_line)
            else:
                processed_lines.append(line)
        return '\n'.join(processed_lines)

    def generate_message_content(
            self,
            template_fields: Dict[str, Any] = None,
            remove_empty_template_field: bool = True,
            **kwargs
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
        # Combine template_fields and kwargs, with kwargs taking precedence
        all_fields = {**(template_fields or {}), **kwargs}

        cleaned_fields = {
            key: str(value) if not isinstance(value, str) else value
            for key, value in all_fields.items()
        }

        if not remove_empty_template_field:
            def replace_placeholder(match):
                placeholder = match.group(1)
                return cleaned_fields.get(placeholder, match.group(0))

            prompt = re.sub(r"\{(\w+)\}", replace_placeholder, self.template)
            return prompt

        def replace_placeholder(match):
            placeholder = match.group(1)
            k = cleaned_fields.get(placeholder, None)
            if k is not None:
                return cleaned_fields.get(placeholder, match.group(0))
            return "__EMPTY_TEMPLATE_FIELD__"

        # Initial placeholder replacement
        prompt = re.sub(r"\{(\w+)\}", replace_placeholder, self.template)

        return self._remove_empty_placeholders(prompt)




