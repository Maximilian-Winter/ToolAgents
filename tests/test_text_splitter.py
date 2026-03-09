import re
import unittest

from ToolAgents.knowledge.text_processing.text_splitter import (
    RecursiveCharacterTextSplitter,
    SimpleTextSplitter,
)


class TestSimpleTextSplitter(unittest.TestCase):
    def test_non_overlapping(self):
        text = 'abcdefghij'
        splitter = SimpleTextSplitter(chunk_size=5, overlap=0)
        expected = ['abcde', 'fghij']
        self.assertEqual(splitter.get_chunks(text), expected)

    def test_overlapping(self):
        text = 'abcdefghij'
        splitter = SimpleTextSplitter(chunk_size=5, overlap=2)
        expected = ['abcde', 'defgh', 'ghij', 'j']
        self.assertEqual(splitter.get_chunks(text), expected)

    def test_chunk_size_greater_than_text(self):
        text = 'abc'
        splitter = SimpleTextSplitter(chunk_size=5, overlap=0)
        self.assertEqual(splitter.get_chunks(text), ['abc'])

    def test_empty_text(self):
        text = ''
        splitter = SimpleTextSplitter(chunk_size=5, overlap=0)
        self.assertEqual(splitter.get_chunks(text), [])

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            SimpleTextSplitter(chunk_size=0, overlap=0)

    def test_invalid_overlap_negative(self):
        with self.assertRaises(ValueError):
            SimpleTextSplitter(chunk_size=5, overlap=-1)

    def test_invalid_overlap_too_large(self):
        with self.assertRaises(ValueError):
            SimpleTextSplitter(chunk_size=5, overlap=5)


class TestRecursiveCharacterTextSplitter(unittest.TestCase):
    def test_no_split_needed(self):
        text = 'Hello world'
        splitter = RecursiveCharacterTextSplitter(
            separators=[' '], chunk_size=20, chunk_overlap=5
        )
        chunks = splitter.get_chunks(text)
        self.assertEqual(chunks, [text])

    def test_split_on_separator(self):
        text = 'a bb ccc dddd'
        splitter = RecursiveCharacterTextSplitter(
            separators=[' '], chunk_size=10, chunk_overlap=2
        )
        self.assertEqual(splitter.get_chunks(text), ['abbcccdddd'])

    def test_fallback_to_fixed_size(self):
        text = 'abcdefghij'
        splitter = RecursiveCharacterTextSplitter(
            separators=[','], chunk_size=3, chunk_overlap=1
        )
        expected = ['abc', 'cde', 'efg', 'ghi', 'ij']
        self.assertEqual(splitter.get_chunks(text), expected)

    def test_keep_separator(self):
        text = 'Hello. World. Test.'
        splitter = RecursiveCharacterTextSplitter(
            separators=['. '], chunk_size=10, chunk_overlap=2, keep_separator=True
        )
        expected = ['Hello. ', 'World. ', 'Test.']
        self.assertEqual(splitter.get_chunks(text), expected)

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            RecursiveCharacterTextSplitter(
                separators=[' '], chunk_size=0, chunk_overlap=0
            )

    def test_invalid_chunk_overlap_negative(self):
        with self.assertRaises(ValueError):
            RecursiveCharacterTextSplitter(
                separators=[' '], chunk_size=5, chunk_overlap=-1
            )

    def test_invalid_chunk_overlap_too_large(self):
        with self.assertRaises(ValueError):
            RecursiveCharacterTextSplitter(
                separators=[' '], chunk_size=5, chunk_overlap=5
            )


if __name__ == '__main__':
    unittest.main()
