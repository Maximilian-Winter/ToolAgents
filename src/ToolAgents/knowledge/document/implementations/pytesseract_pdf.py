from pytesseract import pytesseract

from ToolAgents.knowledge import DocumentGenerator, Document
from ToolAgents.knowledge.document import DocumentProvider
from ToolAgents.knowledge.text_processing.text_splitter import TextSplitter


class PDFOCRProvider(DocumentProvider):

    def __init__(self, folder: str, text_splitter: TextSplitter):
        self.folder = folder
        self.document_generator = DocumentGenerator(text_splitter=text_splitter)


    def get_documents(self, **kwargs) -> list[Document]:
        documents = []
        for text in self.process_pdf(self.folder):
            documents.append(self.document_generator.generate_document(text))

        return documents

    @staticmethod
    def process_page(page):
        # Convert the page to grayscale
        page = page.convert('L')

        # Apply OCR using the preloaded pytesseract
        page_text = pytesseract.image_to_string(page)

        return page_text


    @staticmethod
    def find_files_by_extension(folder_path, extension):
        import os
        """
        Find all files with a specific extension in a given folder.

        Args:
            folder_path (str): Path to the folder to search in
            extension (str): File extension to search for (e.g., '.pdf')

        Returns:
            list: List of full file paths for files with the specified extension
        """
        # Make sure extension starts with a dot
        if not extension.startswith('.'):
            extension = '.' + extension

        matching_files = []

        # Walk through the directory
        try:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(extension.lower()):
                        # Create full file path and add to list
                        full_path = os.path.join(root, file)
                        matching_files.append(full_path)

            return matching_files

        except Exception as e:
            print(f"Error accessing directory: {e}")
            return []

    @staticmethod
    def process_pdf(path):
        results = []
        for file in PDFOCRProvider.find_files_by_extension(path, "pdf"):
            pages = convert_from_path(str(file), dpi=300, fmt='PNG')
            page_texts = Parallel(n_jobs=-1)(delayed(PDFOCRProvider.process_page)(page) for page in pages)
            results.append('\n'.join(page_texts))
        return results


