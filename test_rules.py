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
        self.assertNotEqual(ptr,"foo")
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

    def testEngineArgnames(self):
        argnames = lambda func: Dispatching(func).engine.argnames
        self.assertEqual(
            argnames(lambda a,b,c=None,*d,**e: None), list('abcde')
        )
        self.assertEqual(
            argnames(lambda a,b,c=None,*d: None), list('abcd')
        )
        self.assertEqual(
            argnames(lambda a,b,c=None,**e: None), list('abce')
        )
        self.assertEqual(
            argnames(lambda a,b,c=None: None), list('abc')
        )
        self.assertEqual(
            argnames(lambda a,(b,(c,d)), e: None), list('abcde')
        )










    def testIndexedEngine(self):
        from peak.rules.predicates import IndexedEngine, Comparison
        from peak.rules.criteria import Range, Value, Test, Signature
        from peak.util.assembler import Local
        from peak.util.extremes import Min, Max
        abstract()
        def classify(age): pass
        Dispatching(classify).create_engine(IndexedEngine)
        def setup(r, f):
            when(classify, Signature([Test(Comparison(Local('age')), r)]))(f)
        setup(Range(hi=( 2,-1)), lambda age:"infant")
        setup(Range(hi=(13,-1)), lambda age:"preteen")
        setup(Range(hi=( 5,-1)), lambda age:"preschooler")
        setup(Range(hi=(20,-1)), lambda age:"teenager")
        setup(Range(lo=(20,-1)), lambda age:"adult")
        setup(Range(lo=(55,-1)), lambda age:"senior")
        setup(Value(16), lambda age:"sweet sixteen")

        self.assertEqual(classify(0),"infant")
        self.assertEqual(classify(25),"adult")
        self.assertEqual(classify(17),"teenager")
        self.assertEqual(classify(13),"teenager")
        self.assertEqual(classify(12.99),"preteen")
        self.assertEqual(classify(4),"preschooler")
        self.assertEqual(classify(55),"senior")
        self.assertEqual(classify(54.9),"adult")
        self.assertEqual(classify(14.5),"teenager")
        self.assertEqual(classify(16),"sweet sixteen")
        self.assertEqual(classify(16.5),"teenager")
        self.assertEqual(classify(99),"senior")
        self.assertEqual(classify(Min),"infant")
        self.assertEqual(classify(Max),"senior")









    def testParseInequalities(self):
        from peak.rules.predicates import CriteriaBuilder, Comparison, Truth
        from peak.util.assembler import Compare, Local
        from peak.rules.criteria import Inequality, Test, Value
        from peak.rules.ast_builder import parse_expr
        builder = CriteriaBuilder(
            dict(x=Local('x'), y=Local('y')), locals(), globals(), __builtins__
        )
        def pe(expr):
            return parse_expr(expr, builder)

        x_cmp_y = lambda op, t=True: Test(
            Truth(Compare(Local('x'), ((op, Local('y')),))), Value(t)
        )      
        x,y = Comparison(Local('x')), Comparison(Local('y'))

        for op, mirror_op, not_op, stdop, not_stdop in [
            ('>', '<', '<=','>','<='),
            ('<', '>', '>=','<','>='),
            ('==','==','!=','==','!='),
            ('<>','<>','==','!=','=='),
        ]:
            fwd_sig = Test(x, Inequality(op, 1))
            self.assertEqual(pe('x %s 1' % op), fwd_sig)
            self.assertEqual(pe('1 %s x' % mirror_op), fwd_sig)

            rev_sig = Test(x, Inequality(mirror_op, 1))
            self.assertEqual(pe('x %s 1' % mirror_op), rev_sig)
            self.assertEqual(pe('1 %s x' % op), rev_sig)

            not_sig = Test(x, Inequality(not_op, 1))
            self.assertEqual(pe('not x %s 1' % op), not_sig)
            self.assertEqual(pe('not x %s 1' % not_op), fwd_sig)

            self.assertEqual(pe('x %s y' % op), x_cmp_y(stdop))
            self.assertEqual(pe('x %s y' % not_op), x_cmp_y(not_stdop))

            self.assertEqual(pe('not x %s y' % op),x_cmp_y(stdop,False))
            self.assertEqual(pe('not x %s y' % not_op),x_cmp_y(not_stdop,False))


class MockBuilder:
    def __init__(self, test, expr, cases, remaining, seeds, index=None):
        self.test = test
        self.args = expr, cases, remaining, {}
        self.seeds = seeds
        self.index = index

    def test_func(self, func):
        return func(self, *self.args)

    def build(self, cases, remaining_exprs, memo):
        self.test.failUnless(memo is self.args[-1])
        self.test.assertEqual(self.args[-2], remaining_exprs)
        return cases

    def seed_bits(self, expr, cases):
        self.test.assertEqual(self.args[1], cases)
        if self.index is not None:
            return self.index.seed_bits(cases)
        return self.seeds
        
    def reseed(self, expr, criterion):
        self.test.assertEqual(self.args[0], expr)
        self.index.reseed(criterion)

















class NodeBuildingTests(unittest.TestCase):

    def build(self, func, dontcare, seeds, index=None):
        seedbits = dontcare, seeds
        builder = MockBuilder(
            self, 'expr', 'cases', 'remaining', seedbits, index
        )
        return builder.test_func(func)

    def testTruthNode(self):
        from peak.rules.predicates import truth_node
        node = self.build(truth_node, 27,
            {(True,0): (128,0), (False,0): (64,0)})
        self.assertEqual(node, (27|128, 27|64))

    def testIdentityNode(self):
        from peak.rules.predicates import identity_node
        node = self.build(identity_node, 27,
            {9127: (128,0), 6499: (64,0), None: (0,0)})
        self.assertEqual(node, {None:27, 9127:27|128, 6499: 27|64})

    def testRangeNode(self):
        from peak.rules.indexing import BitmapIndex, to_bits
        from peak.rules.predicates import range_node
        from peak.rules.criteria import Range, Value, Min, Max
        ind = BitmapIndex(self, 'expr')
        ind.add_case(0, Value(19))
        ind.add_case(1, Value(23))
        ind.add_case(2, Value(23, False))
        ind.add_case(3, Range(lo=(57,1)))
        ind.add_case(4, Range(lo=(57,-1)))
        dontcare, seeds = ind.seed_bits(to_bits(range(6)))
        exact, ranges = self.build(range_node, dontcare, seeds)
        self.assertEqual(exact,
            {19:to_bits([0, 2, 5]), 23:to_bits([1,5]), 57:to_bits([2,4,5])})
        self.assertEqual(ranges,
            [((Min,57), to_bits([2,5])),  ((57,Max), to_bits([2,3,4,5]))])




    def testClassNode(self):
        from peak.rules.indexing import BitmapIndex, to_bits
        from peak.rules.predicates import class_node
        from peak.rules.criteria import Class, Classes
        from types import InstanceType
        ind = BitmapIndex(self, 'expr')
        class a: pass
        class b: pass
        class c(a,b): pass
        class x(a,b,object): pass

        ind.add_case(0, Class(InstanceType))
        ind.add_case(1, Classes([Class(a), Class(b), Class(c,False)]))
        ind.add_case(2, Class(object))
        ind.add_case(3, Classes([Class(a), Class(b)]))
        ind.add_case(4, Class(a))
        ind.selectivity(range(6))
        cases = to_bits(range(6))
        builder = MockBuilder(
            self, 'expr', cases, 'remaining', ind.seed_bits(cases), ind
        )
        cache, lookup = builder.test_func(class_node,)
        self.assertEqual(cache, {})
        data = [
            (object, to_bits([2,5])),
            (InstanceType, to_bits([0,2,5])),
            (a, to_bits([0,2,4,5])),
            (b, to_bits([0,2,5])),
            (c, to_bits([0,2,3,4,5])),
            (x, to_bits([0,1,2,3,4,5]))
        ]
        for k, v in data:
            self.assertEqual(lookup(k), v)
        self.assertEqual(cache, dict(data))
        
        





def additional_tests():
    import doctest
    return doctest.DocFileSuite(
        'DESIGN.txt', 'Indexing.txt', 'AST-Builder.txt',
        'Code-Generation.txt', 'Criteria.txt', 'Predicates.txt',
        optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE,
    )


































