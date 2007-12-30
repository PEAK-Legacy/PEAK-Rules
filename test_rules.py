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

    def testAbstract(self):
        def f1(x,y=None):
            raise AssertionError("Should never get here")
        d = Dispatching(f1)
        log = []
        d.rules.default_action = lambda *args: log.append(args)
        f1 = abstract(f1)
        f1(27,42)
        self.assertEqual(log, [(27,42)])
        when(f1, ())(lambda *args: 99)
        self.assertRaises(AssertionError, abstract, f1)

    def testAbstractRegeneration(self):
        def f1(x,y=None):
            raise AssertionError("Should never get here")
        d = Dispatching(f1)
        log = []
        d.rules.default_action = lambda *args: log.append(args)
        d.request_regeneration()
        f1 = abstract(f1)
        self.assertNotEqual(d.backup, f1.func_code)
        self.assertEqual(f1.func_code, d._regen)
        f1.func_code = d.backup
        f1(27,42)
        self.assertEqual(log, [(27,42)])
        
    def testCreateEngine(self):
        def f1(x,y=None):
            raise AssertionError("Should never get here")
        d = Dispatching(f1)
        old_engine = d.engine
        self.assertEqual(d.rules.listeners, [old_engine])
        from peak.rules.core import TypeEngine
        class MyEngine(TypeEngine): pass
        d.create_engine(MyEngine)
        new_engine = d.engine
        self.assertNotEqual(new_engine, old_engine)
        self.failUnless(isinstance(new_engine, MyEngine))
        self.assertEqual(d.rules.listeners, [new_engine])


    def testIndexClassicMRO(self):
        class MyEngine: pass
        eng = MyEngine()
        from peak.rules.indexing import BitmapIndex
        from peak.rules.criteria import Class
        from types import InstanceType
        ind = BitmapIndex(eng, 'classes')
        ind.add_case(0, Class(MyEngine))
        ind.add_case(1, Class(object))
        self.assertEqual(
            dict(ind.expanded_sets()),
            {MyEngine: [[0],[]], InstanceType: [[],[]], object: [[1],[]]}
        )



























        
def additional_tests():
    import doctest
    return doctest.DocFileSuite(
        'DESIGN.txt', 'Indexing.txt', 'AST-Builder.txt',
        'Code-Generation.txt', 'Criteria.txt',
        optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE,
    )


































