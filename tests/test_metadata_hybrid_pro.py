import unittest

from src.train_eval_hybrid_pro import clean_text as hybrid_clean_text
from src.train_eval_hybrid_pro import source_domain as hybrid_source_domain
from src.train_eval_hybrid_pro import tweet_count as hybrid_tweet_count
from src.train_eval_metadata_pro import source_domain as metadata_source_domain
from src.train_eval_metadata_pro import tweet_count as metadata_tweet_count


class TestMetadataHybridProUtilities(unittest.TestCase):
    def test_tweet_count_handles_tabs_and_empty(self):
        self.assertEqual(metadata_tweet_count("1\t2\t3"), 3)
        self.assertEqual(metadata_tweet_count(""), 0)
        self.assertEqual(hybrid_tweet_count(None), 0)

    def test_source_domain_handles_missing(self):
        self.assertEqual(metadata_source_domain("https://www.bbc.com/news"), "bbc.com")
        self.assertEqual(hybrid_source_domain(""), "unknown")

    def test_hybrid_clean_text_basic(self):
        got = hybrid_clean_text("Hello!!! Visit http://x.com")
        self.assertEqual(got, "hello visit")


if __name__ == "__main__":
    unittest.main()
