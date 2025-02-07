import re
import unittest

from ToolAgents.utilities import SimpleTextSplitter, RecursiveCharacterTextSplitter


# Now, let's write our tests.

class TestSimpleTextSplitter(unittest.TestCase):
    def test_non_overlapping(self):
        # Test splitting without overlap.
        text = "abcdefghij"
        splitter = SimpleTextSplitter(text, chunk_size=5, overlap=0)
        expected = ["abcde", "fghij"]
        self.assertEqual(splitter.get_chunks(), expected)

    def test_overlapping(self):
        # Test splitting with an overlap.
        text = "abcdefghij"
        splitter = SimpleTextSplitter(text, chunk_size=5, overlap=2)
        # Expected behavior: start=0 -> "abcde", then start=3 -> "defgh", then start=6 -> "ghij", then start=9 -> "j"
        expected = ["abcde", "defgh", "ghij", "j"]
        self.assertEqual(splitter.get_chunks(), expected)

    def test_chunk_size_greater_than_text(self):
        # When the chunk size is larger than the text, we should get back the original text.
        text = "abc"
        splitter = SimpleTextSplitter(text, chunk_size=5, overlap=0)
        self.assertEqual(splitter.get_chunks(), ["abc"])

    def test_empty_text(self):
        # Splitting an empty string should return an empty list.
        text = ""
        splitter = SimpleTextSplitter(text, chunk_size=5, overlap=0)
        self.assertEqual(splitter.get_chunks(), [])

    def test_invalid_chunk_size(self):
        # A non-positive chunk_size should raise a ValueError.
        with self.assertRaises(ValueError):
            SimpleTextSplitter("abc", chunk_size=0, overlap=0)

    def test_invalid_overlap_negative(self):
        # Negative overlap should raise a ValueError.
        with self.assertRaises(ValueError):
            SimpleTextSplitter("abc", chunk_size=5, overlap=-1)

    def test_invalid_overlap_too_large(self):
        # Overlap equal to (or greater than) chunk_size should raise a ValueError.
        with self.assertRaises(ValueError):
            SimpleTextSplitter("abc", chunk_size=5, overlap=5)


class TestRecursiveCharacterTextSplitter(unittest.TestCase):
    def test_no_split_needed(self):
        # If the text is short, no splitting should occur.
        text = "Hello world"
        splitter = RecursiveCharacterTextSplitter(separators=[" "], chunk_size=20, chunk_overlap=5)
        chunks = splitter.split_text(text)
        self.assertEqual(chunks, [text])

    def test_split_on_separator(self):
        # Test splitting on a space.
        text = "a bb ccc dddd"
        splitter = RecursiveCharacterTextSplitter(separators=[" "], chunk_size=10, chunk_overlap=2)
        # The text.split(" ") gives ["a", "bb", "ccc", "dddd"]
        # Merging these pieces:
        #   current_chunk = "a"
        #   "a" + "bb" => "abb" (len 3)
        #   "abb" + "ccc" => "abbccc" (len 6)
        #   "abbccc" + "dddd" => "abbcccdddd" (len 10, exactly full)
        self.assertEqual(splitter.split_text(text), ["abbcccdddd"])

    def test_fallback_to_fixed_size(self):
        # If the separator isn't found, it should fall back to fixed-size splitting.
        text = "abcdefghij"
        # Using a separator that doesn't exist (",") forces the fallback.
        splitter = RecursiveCharacterTextSplitter(separators=[","], chunk_size=3, chunk_overlap=1)
        # _split_into_fixed_size: step = 3 - 1 = 2, so:
        #   text[0:3] -> "abc"
        #   text[2:5] -> "cde"
        #   text[4:7] -> "efg"
        #   text[6:9] -> "ghi"
        #   text[8:11] -> "ij"
        expected = ["abc", "cde", "efg", "ghi", "ij"]
        self.assertEqual(splitter.split_text(text), expected)

    def test_keep_separator(self):
        # Test splitting when keep_separator is True.
        text = "Hello. World. Test."
        splitter = RecursiveCharacterTextSplitter(
            separators=[". "], chunk_size=10, chunk_overlap=2, keep_separator=True
        )
        # Using re.split with the separator ". " on the text should yield:
        #   ["Hello", ". ", "World", ". ", "Test."]
        # Then the logic merges the separator back into the preceding piece:
        #   -> ["Hello. ", "World. ", "Test."]
        # Finally, _merge_pieces will try to merge chunks but here each piece is too long to combine.
        expected = ["Hello. ", "World. ", "Test."]
        self.assertEqual(splitter.split_text(text), expected)

    def test_invalid_chunk_size(self):
        # A non-positive chunk_size should raise a ValueError.
        with self.assertRaises(ValueError):
            RecursiveCharacterTextSplitter(separators=[" "], chunk_size=0, chunk_overlap=0)

    def test_invalid_chunk_overlap_negative(self):
        # Negative chunk_overlap should raise a ValueError.
        with self.assertRaises(ValueError):
            RecursiveCharacterTextSplitter(separators=[" "], chunk_size=5, chunk_overlap=-1)

    def test_invalid_chunk_overlap_too_large(self):
        # chunk_overlap equal to (or greater than) chunk_size should raise a ValueError.
        with self.assertRaises(ValueError):
            RecursiveCharacterTextSplitter(separators=[" "], chunk_size=5, chunk_overlap=5)


if __name__ == '__main__':
    unittest.main()
