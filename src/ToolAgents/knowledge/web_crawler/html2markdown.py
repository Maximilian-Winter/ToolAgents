from bs4 import BeautifulSoup
import re
from typing import Optional, Union
from pathlib import Path


class HTML2Markdown:
    def __init__(self):
        self.list_depth = 0
        self.in_code_block = False

    def convert(self, html: Union[str, Path]) -> str:
        """Convert HTML to Markdown.

        Args:
            html: HTML content as string or path to HTML file

        Returns:
            Converted Markdown text
        """
        if isinstance(html, Path):
            html = html.read_text(encoding='utf-8')

        # Parse HTML and get body content only
        soup = BeautifulSoup(html, 'html5lib')
        body = soup.find('body')

        # Remove script and style elements
        for element in (body or soup).find_all(['script', 'style', 'nav', 'header']):
            element.decompose()

        return self._convert_element(body or soup)

    def _convert_element(self, element) -> str:
        """Recursively convert HTML elements to Markdown."""
        if element.name is None:
            return self._clean_text(element.string or '')

        markdown = []
        name = element.name.lower()

        if name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = int(name[1])
            markdown.append(f"{'#' * level} {self._clean_text(element.get_text())}\n\n")

        elif name == 'p':
            text = self._convert_text_with_inline_elements(element)
            if text:
                markdown.append(f"{text}\n\n")

        elif name == 'br':
            markdown.append("\n")

        elif name == 'hr':
            markdown.append("\n---\n\n")

        elif name == 'blockquote':
            quote = self._convert_text_with_inline_elements(element)
            # Prefix each line with '> '
            quote_lines = quote.splitlines()
            markdown.append('\n'.join('> ' + line for line in quote_lines) + '\n\n')

        elif name == 'a':
            url = element.get('href', '')
            text = self._clean_text(element.get_text())
            if url and text:
                markdown.append(f"[{text}]({url})")

        elif name == 'img':
            alt = element.get('alt', '')
            src = element.get('src', '')
            if src:
                markdown.append(f"![{alt}]({src})\n\n")

        elif name in ['ul', 'ol']:
            self.list_depth += 1
            # Process direct li children only
            for i, item in enumerate(element.find_all('li', recursive=False)):
                prefix = '  ' * (self.list_depth - 1)
                marker = '*' if name == 'ul' else f"{i + 1}."
                # Recursively convert the entire li to capture nested lists or block elements
                li_content = self._convert_element(item).rstrip()
                markdown.append(f"{prefix}{marker} {li_content}\n")
            self.list_depth -= 1
            if self.list_depth == 0:
                markdown.append('\n')

        elif name == 'li':
            # For list items, process children recursively to support nested lists
            for child in element.children:
                markdown.append(self._convert_element(child))

        elif name == 'pre':
            self.in_code_block = True
            code = element.find('code')
            if code:
                # Try to detect language from classes, e.g., class="language-python"
                lang = ''
                classes = code.get('class', [])
                for cls in classes:
                    if cls.startswith('language-'):
                        lang = cls.replace('language-', '')
                        break
                code_text = code.get_text().strip()
                markdown.append(f"```{lang}\n{code_text}\n```\n\n")
            else:
                code_text = element.get_text().strip()
                markdown.append(f"```\n{code_text}\n```\n\n")
            self.in_code_block = False

        elif name == 'code' and not self.in_code_block:
            markdown.append(f"`{element.get_text()}`")

        elif name in ['strong', 'b']:
            markdown.append(f"**{self._clean_text(element.get_text())}**")

        elif name in ['em', 'i']:
            markdown.append(f"*{self._clean_text(element.get_text())}*")

        elif name == 'table':
            markdown.extend(self._convert_table(element))

        else:
            # Default: recursively process child elements
            for child in element.children:
                if isinstance(child, str):
                    markdown.append(self._clean_text(child))
                else:
                    markdown.append(self._convert_element(child))

        return ''.join(markdown)

    def _convert_text_with_inline_elements(self, element) -> str:
        """Convert text while preserving inline formatting."""
        result = []
        for child in element.children:
            if isinstance(child, str):
                text = self._clean_text(child)
                result.append(text)
            elif child.name in ['strong', 'b']:
                text = self._clean_text(child.get_text())
                result.append(f"**{text}**")
            elif child.name in ['em', 'i']:
                text = self._clean_text(child.get_text())
                result.append(f"*{text}*")
            elif child.name == 'code':
                text = child.get_text()
                result.append(f"`{text}`")
            elif child.name == 'a':
                url = child.get('href', '')
                text = self._clean_text(child.get_text())
                result.append(f"[{text}]({url})")
            else:
                text = self._convert_element(child)
                result.append(text)
        return ''.join(result).strip()

    def _convert_table(self, table) -> list:
        """Convert HTML table to Markdown table format."""
        markdown = []
        headers = []
        rows = []

        # Get headers
        for th in table.find_all('th'):
            headers.append(self._clean_text(th.get_text()))

        # If no headers found, try first row as header
        if not headers and table.find('tr'):
            first_row = table.find('tr')
            headers = [self._clean_text(td.get_text()) for td in first_row.find_all('td')]
            rows = table.find_all('tr')[1:]
        else:
            rows = table.find_all('tr')

        if not headers:
            return []

        # Add headers and separator row
        markdown.append('| ' + ' | '.join(headers) + ' |\n')
        markdown.append('| ' + ' | '.join(['---'] * len(headers)) + ' |\n')

        # Add rows
        for row in rows:
            cells = row.find_all('td')
            if cells:
                row_data = [self._clean_text(cell.get_text()) for cell in cells]
                markdown.append('| ' + ' | '.join(row_data) + ' |\n')

        markdown.append('\n\n')  # Extra newline for proper separation
        return markdown

    def _clean_text(self, text: Optional[str]) -> str:
        """Clean and normalize text content.

        Skips escaping when inside a code block.
        """
        if text is None:
            return ''

        text = text.strip()
        if not self.in_code_block:
            # Normalize whitespace and escape special markdown characters
            text = re.sub(r'\s+', ' ', text)
            special_chars = ['[', ']', '#', '`']
            for char in special_chars:
                text = text.replace(char, '\\' + char)
        return text


def convert_file(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> None:
    """Convert HTML file to Markdown file.

    Args:
        input_path: Path to input HTML file
        output_path: Path to output Markdown file (optional)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix('.md')
    else:
        output_path = Path(output_path)
    converter = HTML2Markdown()
    markdown = converter.convert(input_path)

    output_path.write_text(markdown, encoding='utf-8')


if __name__ == '__main__':
    import sys

    input_file = "test.html"
    output_file = "test.md"

    try:
        convert_file(input_file, output_file)
        print(f"Conversion complete: {output_file or input_file[:-5] + '.md'}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
