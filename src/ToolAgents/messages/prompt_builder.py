import abc
import os


class PromptPart(abc.ABC):

    @abc.abstractmethod
    def build(self) -> str:
        pass

class PromptVar(PromptPart):

    def __init__(self, var = None) -> None:
        self.var = var

    def set_var(self, var) -> None:
        self.var = var

    def get_var(self):
        return self.var

    def build(self) -> str:
        return str(self.var)

class PromptLine(PromptPart):

    def __init__(self) -> None:
        super().__init__()
        self.line = []

    def add_characters(self, character, n=1):
        for _ in range(n):
            self.line.append(character)
        return self

    def add_part(self, part):
        self.line.append(part)
        return self

    def add_text(self, text):
        self.line.append(text)
        return self

    def build(self):
        section = ""
        for part in self.line:
            if issubclass(type(part), PromptPart):
                section += part.build()
            else:
                section += str(part)
        return section


class PromptBuilder:

    def __init__(self) -> None:
        super().__init__()
        self.content = []

    def add_prompt_part(self, part: PromptPart):
        self.content.append(part)
        return self

    def add_text(self, text):
        self.content.append(text)
        return self

    def add_file_content(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r') as file:
            content = file.read()
        self.content.append(content)
        return self

    def add_empty_line(self, n=1):
        for _ in range(n):
            self.content.append("")
        return self

    def add_numbered_list(self, items):
        numbered_list = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
        self.content.append(numbered_list)
        return self

    def add_bullet_list(self, items):
        bullet_list = "\n".join(f"• {item}" for item in items)
        self.content.append(bullet_list)
        return self

    def add_code_block(self, code, language=""):
        formatted_code = f"```{language}\n{code}\n```"
        self.content.append(formatted_code)
        return self

    def add_separator(self, char="-", length=40):
        separator = char * length
        self.content.append(separator)
        return self

    def build(self):
        section = ""
        for part in self.content:
            if issubclass(type(part), PromptPart):
                section += part.build()
            else:
                section += str(part)
            section += "\n"
        return section



if __name__ == "__main__":
    builder = PromptBuilder()
    builder.add_text("You are helpful assistant.").add_separator('#', 15)
    line = PromptLine()
    builder.add_prompt_part(line)
    var = PromptVar(42)
    line.add_characters('#', 10).add_characters(' ').add_part(var).add_characters(' ').add_characters('#', 10)
    var.set_var(24)
    print(line.build())
    print(builder.build())
