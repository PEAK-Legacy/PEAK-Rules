import unittest
from peak.rules import *

x2 = lambda a: a*2
x3 = lambda next_method, a: next_method(a)*3

class TypeEngineTests(unittest.TestCase):

    def testIntraSignatureCombinationAndRemoval(self):
        abstract()
        def f(a):
            """blah"""

        rx2 = Rule(x2,(int,), Method)
        rx3 = Rule(x3,(int,), Around)

        rules_for(f).add(rx2)
        self.assertEqual(f(1), 2)

        rules_for(f).add(rx3)
        self.assertEqual(f(1), 6)

        rules_for(f).remove(rx3)
        self.assertEqual(f(1), 2)

    def testAroundDecoratorAndRetroactiveCombining(self):
        def f(a):
            return a

        self.assertEqual(f(1), 1)
        self.assertEqual(f('x'), 'x')

        when(f, (int,))(x2)
        self.assertEqual(f(1), 2)
        self.assertEqual(f('x'), 'x')

        around(f, (int,))(lambda a:42)
        self.assertEqual(f(1), 42)
        self.assertEqual(f('x'), 'x')


def additional_tests():
    import doctest
    return doctest.DocFileSuite(
        'DESIGN.txt', 'Indexing.txt', 'AST-Builder.txt',
        'Code-Generation.txt', 'Aspects.txt',
        optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE,
    )


































