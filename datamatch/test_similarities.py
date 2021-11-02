import unittest
from datetime import date

from datamatch.similarities import (
    StringSimilarity, DateSimilarity, JaroWinklerSimilarity, AbsoluteNumericalSimilarity, RelativeNumericalSimilarity
)


class TestStringSimilarity(unittest.TestCase):
    def test_sim(self):
        obj = StringSimilarity()
        self.assertEqual(obj.sim("abc", "abc"), 1)
        self.assertEqual(obj.sim("abc", "123"), 0)
        self.assertEqual(obj.sim("abce", "abcd"), 0.75)
        self.assertEqual(obj.sim("thang", "thăng"), 1)


class TestJaroWinklerSimilarity(unittest.TestCase):
    def test_sim(self):
        obj = JaroWinklerSimilarity(0.2)
        self.assertEqual(obj.sim("abc", "abc"), 1)
        self.assertEqual(obj.sim("abc", "123"), 0)
        self.assertEqual(obj.sim("abce", "abcd"), 0.9333333333333333)
        self.assertEqual(obj.sim("wbcd", "abcd"), 0.8333333333333334)


class TestDateSimilarity(unittest.TestCase):
    def test_sim(self):
        obj = DateSimilarity()
        self.assertEqual(obj.sim(date(2000, 10, 11), date(2000, 10, 11)), 1)
        # within 30 days difference
        self.assertEqual(obj.sim(date(2000, 10, 11), date(2000, 10, 5)), 0.8)
        self.assertEqual(
            obj.sim(date(2000, 10, 11), date(2000, 11, 5)), 0.16666666666666663)
        # completely different days
        self.assertEqual(obj.sim(date(2000, 10, 11), date(2001, 3, 15)), 0)
        # day & month is swapped
        self.assertEqual(obj.sim(date(2000, 9, 11), date(2000, 11, 9)), 0.5)
        # same year and day but month is different
        self.assertEqual(obj.sim(date(2000, 3, 20), date(2000, 8, 20)), 0.875)


class TestAbsoluteNumericalSimilarity(unittest.TestCase):
    def test_sim(self):
        obj = AbsoluteNumericalSimilarity(10)
        self.assertEqual(obj.sim(10, 10), 1)
        self.assertEqual(obj.sim(8.9, 8.9), 1)
        self.assertEqual(obj.sim(10, 5), 0.5)
        self.assertEqual(obj.sim(10, 15), 0.5)
        self.assertEqual(obj.sim(8.2, 3.1), 0.49)
        self.assertEqual(obj.sim(40, 10), 0)


class TestRelativeNumericalSimilarity(unittest.TestCase):
    def test_sim(self):
        obj = RelativeNumericalSimilarity(30)
        self.assertEqual(obj.sim(10000, 10000), 1)
        self.assertEqual(obj.sim(8.9, 8.9), 1)
        self.assertEqual(obj.sim(10000, 8500), 0.5)
        self.assertEqual(obj.sim(8500, 10000), 0.5)
        self.assertEqual(obj.sim(8.2, 3.1), 0)
        self.assertEqual(obj.sim(10000, 7000), 0)
