import unittest

from training.progress import resume_progress


class TrainingResumeTests(unittest.TestCase):
    def test_keeps_loaded_progress_for_same_run_resume(self):
        self.assertEqual(resume_progress(2321, 2433744896, reset=False), (2321, 2433744896))

    def test_resets_loaded_progress_for_continuation_run(self):
        self.assertEqual(resume_progress(2321, 2433744896, reset=True), (0, 0))


if __name__ == "__main__":
    unittest.main()
