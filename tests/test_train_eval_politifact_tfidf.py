import unittest

from src.train_eval_politifact_tfidf import (
    choose_threshold_by_f1,
    clean_text,
    evaluate_at_threshold,
    source_domain,
)


class TestTrainEvalUtilities(unittest.TestCase):
    def test_clean_text_normalizes(self):
        text = "Breaking: Visit https://example.com NOW!!!"
        got = clean_text(text)
        self.assertEqual(got, "breaking visit now")

    def test_source_domain_parsing(self):
        self.assertEqual(source_domain("https://www.nytimes.com/a/b"), "nytimes.com")
        self.assertEqual(source_domain(""), "unknown")

    def test_threshold_metrics(self):
        y_true = [0, 1, 1, 0]
        y_prob = [0.1, 0.7, 0.4, 0.8]
        out = evaluate_at_threshold(y_true, y_prob, threshold=0.5)
        self.assertAlmostEqual(out["precision"], 0.5)
        self.assertAlmostEqual(out["recall"], 0.5)
        self.assertEqual(out["tp"], 1)
        self.assertEqual(out["fp"], 1)

    def test_choose_threshold(self):
        y_true = [0, 0, 1, 1]
        y_prob = [0.2, 0.3, 0.6, 0.7]
        best, table = choose_threshold_by_f1(y_true, y_prob)
        self.assertTrue(0.1 <= best <= 0.9)
        self.assertIn("f1", table.columns)


if __name__ == "__main__":
    unittest.main()
