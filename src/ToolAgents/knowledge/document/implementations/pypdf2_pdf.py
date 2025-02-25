import os
from typing import List

import PyPDF2

from ToolAgents.knowledge import DocumentGenerator, Document
from ToolAgents.knowledge.document import DocumentProvider
from ToolAgents.knowledge.text_processing.text_splitter import TextSplitter


class PDFProvider(DocumentProvider):
    """
    A document provider that extracts text directly from PDF files without using OCR.
    """

    def __init__(self, folder: str, text_splitter: TextSplitter):
        """
        Initialize the PDF provider.

        Args:
            folder (str): Path to the folder containing PDF files
            text_splitter (TextSplitter): Text splitter instance for chunking documents
        """
        self.folder = folder
        self.document_generator = DocumentGenerator(text_splitter=text_splitter)

    def get_documents(self, **kwargs) -> List[Document]:
        """
        Get documents from PDF files in the specified folder.

        Returns:
            List[Document]: List of processed documents
        """
        documents = []
        for text in self._process_pdf_files(self.folder):
            documents.append(self.document_generator.generate_document(text))
        return documents

    @staticmethod
    def _find_pdf_files(folder_path: str) -> List[str]:
        """
        Find all PDF files in the given folder path.

        Args:
            folder_path (str): Path to search for PDF files

        Returns:
            List[str]: List of PDF file paths
        """
        pdf_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files

    def _process_pdf_files(self, folder_path: str) -> List[str]:
        """
        Process all PDF files in the given folder and extract their text content.

        Args:
            folder_path (str): Path to the folder containing PDF files

        Returns:
            List[str]: List of extracted text content from PDF files
        """
        results = []
        pdf_files = self._find_pdf_files(folder_path)

        for pdf_path in pdf_files:
            try:
                with open(pdf_path, 'rb') as file:
                    # Create PDF reader object
                    pdf_reader = PyPDF2.PdfReader(file)

                    # Extract text from all pages
                    text_content = []
                    for page in pdf_reader.pages:
                        text_content.append(page.extract_text())

                    # Join all pages with newlines
                    full_text = '\n'.join(text_content)
                    results.append(full_text)

            except Exception as e:
                print(f"Error processing PDF file {pdf_path}: {str(e)}")
                continue

        return results

