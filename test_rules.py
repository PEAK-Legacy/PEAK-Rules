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


class MiscTests(unittest.TestCase):

    def testPointers(self):
        from peak.rules.indexing import IsObject
        from sys import maxint
        anOb = object()
        ptr = IsObject(anOb)
        self.assertEqual(id(anOb)&maxint,ptr)
        self.assertEqual(hash(id(anOb)&maxint),hash(ptr))

        self.assertEqual(ptr.match, True)
        self.assertEqual(IsObject(anOb, False).match, False)
        self.assertNotEqual(IsObject(anOb, False), ptr)

        class X: pass
        anOb = X()
        ptr = IsObject(anOb)
        oid = id(anOb)&maxint
        self.assertEqual(oid,ptr)
        self.assertEqual(hash(oid),hash(ptr))
        del anOb
        #self.assertNotEqual(oid,ptr)
        self.assertEqual(ptr,ptr)
        self.assertEqual(hash(oid),hash(ptr))

    def testRuleSetReentrance(self):
        from peak.rules.core import Rule, RuleSet
        rs = RuleSet()
        log = []
        class MyListener:
            def actions_changed(self, added, removed):
                log.append(1)
                if self is ml1:
                    rs.unsubscribe(ml2)
        ml1, ml2 = MyListener(), MyListener()
        rs.subscribe(ml1)
        rs.subscribe(ml2)
        self.assertEqual(log, [])
        rs.add(Rule(lambda:None))
        self.assertEqual(log, [1, 1])

def additional_tests():
    import doctest
    return doctest.DocFileSuite(
        'DESIGN.txt', 'Indexing.txt', 'AST-Builder.txt',
        'Code-Generation.txt', 'Criteria.txt',
        optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE,
    )


































