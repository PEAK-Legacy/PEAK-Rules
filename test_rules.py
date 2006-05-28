import unittest
from peak.rules.framework import *

x2 = lambda a: a*2
x3 = lambda next_method, a: next_method(a)*3

class TypeEngineTests(unittest.TestCase):

    def testIntraSignatureCombinationAndRemoval(self):
        abstract()
        def f(a):
            """blah"""

        rx2 = Rule(x2,(int,), Method)
        rx3 = Rule(x3,(int,), Around)

        f.__rules__.add(rx2)
        self.assertEqual(f(1), 2)

        f.__rules__.add(rx3)
        self.assertEqual(f(1), 6)

        f.__rules__.remove(rx3)
        self.assertEqual(f(1), 2)

    def testAroundDecorator(self):
        abstract()
        def f(a):
            """blah"""

        when(f, (int,))(x2)
        self.assertEqual(f(1), 2)

        around(f, (int,))(lambda a:42)
        self.assertEqual(f(1), 42)






def additional_tests():
    import doctest
    return doctest.DocFileSuite(
        'framework.txt', package='peak.rules',
        optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE,
    )



































