import unittest

from mclust.Main import dummy


class TestDummyMethods(unittest.TestCase):

    def test_dummy(self):
        self.assertEqual(dummy(), 1)


if __name__ == '__main__':
    unittest.main()
