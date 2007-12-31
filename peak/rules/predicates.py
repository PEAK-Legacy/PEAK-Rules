from peak.util.assembler import *
from core import *
from core import class_or_type_of
from criteria import *
from indexing import *
from codegen import SMIGenerator, ExprBuilder, Getitem, IfElse
from peak.util.decorators import decorate, synchronized
from types import InstanceType, ClassType
from ast_builder import build

__all__ = [
    'IsInstance', 'IsSubclass', 'Truth', 'Identity', 'Comparison',
    'IndexedEngine', 'predicate_node_for',
]

abstract()
def predicate_node_for(builder, expr, cases, remaining_exprs, memo):
    """Return a dispatch tree node argument appropriate for the expr"""

def value_check(val, (exact, ranges)):
    if val in exact:
        return exact[val]
    lo = 0
    hi = len(ranges)
    while lo<hi:
        mid = (lo+hi)//2
        (tl,th), node = ranges[mid]
        if val<tl:
            hi = mid
        elif val>th:
            lo = mid+1
        else:
            return node
    raise AssertionError("Should never get here")

nodetype()
def IsInstance(expr, code=None):
    if code is None: return expr,
    return IsSubclass(class_or_type_of(expr), code)


nodetype()
def IsSubclass(expr, code=None):
    if code is None: return expr,
    unpack = lambda c: c.UNPACK_SEQUENCE(2)
    code(
        expr, TryExcept(
            Suite([
                Code.DUP_TOP, SMIGenerator.ARG, unpack, Code.ROT_THREE,
                Code.POP_TOP, Code.BINARY_SUBSCR, Code.ROT_TWO, Code.POP_TOP
            ]), [(Const(KeyError), Suite([
                SMIGenerator.ARG, unpack, Code.POP_TOP, Call(Code.ROT_TWO, (Pass,)),
            ]))]
        )
    )

nodetype()
def Identity(expr, code=None):
    if code is None: return expr,
    code(
        Call(Const(id), (expr,), fold=False),
        IfElse(
            Getitem(SMIGenerator.ARG, Code.ROT_TWO),
            Compare(Code.DUP_TOP, [('in', SMIGenerator.ARG)]),
            Suite([Code.POP_TOP, Getitem(SMIGenerator.ARG, None)])
        )
    )

nodetype()
def Comparison(expr, code=None):
    if code is None: return expr,
    code.LOAD_CONST(value_check)
    Call(Pass, (expr, SMIGenerator.ARG), code=code)

nodetype()
def Truth(expr, code=None):
    if code is None: return expr,
    skip = Label()
    code(SMIGenerator.ARG); code.UNPACK_SEQUENCE(2)
    code(expr, skip.JUMP_IF_TRUE, Code.ROT_THREE, skip, Code.POP_TOP,
         Code.ROT_TWO, Code.POP_TOP)

class CriteriaBuilder:
    simplify_comparisons = True
    mode = True

    def __init__(self, arguments, *namespaces):
        self.expr_builder = ExprBuilder(arguments, *namespaces)

    def mkOp(name):
        op = getattr(ExprBuilder,name)
        def method(self, *args):
            return expressionSignature(op(self.expr_builder, *args), self.mode)
        return method

    for opname in dir(ExprBuilder):
        if opname[0].isalpha() and opname[0]==opname[0].upper():
            locals()[opname] = mkOp(opname)

    def Not(self,expr):
        try:
            self.__class__ = NotBuilder
            return build(self, expr)
        finally:
            self.__class__ = CriteriaBuilder

    _mirror_ops = {
        '>': '<', '>=': '<=', '=>':'<=',
        '<': '>', '<=': '>=', '=<':'>=',
        '<>': '<>', '!=': '<>', '==':'==',
        'is': 'is', 'is not': 'is not'
    }

    _rev_ops = {
        '>': '<=', '>=': '<', '=>': '<',
        '<': '>=', '<=': '>', '=<': '>',
        '<>': '==', '!=': '==', '==':'!=',
        'in': 'not in', 'not in': 'in',
        'is': 'is not', 'is not': 'is'
    }



    def Compare(self, initExpr, ((op,other),)):
        old_op = op
        left = build(self.expr_builder, initExpr)
        right = build(self.expr_builder, other)

        if isinstance(left,Const) and op in self._mirror_ops:
            left, right, op = right, left, self._mirror_ops[op]

        if isinstance(right,Const):
            if not self.mode:
                op = self._rev_ops[op]

            if op=='in' or op=='not in':
                cond = compileIn(left, right.value, op=='in')
                if cond is not None:
                    return cond
            else:
                if op=='is' or op=='is not':
                    #if right.value is None:     # XXX this should be smarter
                    #    left = IsInstance(left)
                    #    right = Class(NoneType)
                    #else:
                        left = Identity(left)
                        right = IsObject(right.value, op=='is')
                else:
                    left = Comparison(left)
                    right = Inequality(op, right.value)                    
                return Test(left, right)

        # Both sides involve variables or an un-optimizable constant,
        #  so it's a generic boolean criterion  :(
        return expressionSignature(
            self.expr_builder.Compare(initExpr, ((old_op,other),)), self.mode
        )

    def And(self, items):
        return reduce(intersect, [build(self,expr) for expr in items])

    def Or(self, items):
        return Disjunction([build(self,expr) for expr in items])

class NotBuilder(CriteriaBuilder):
    mode = False

    def Not(self,expr):
        try:
            self.__class__ = CriteriaBuilder
            return build(self,expr)
        finally:
            self.__class__ = NotBuilder

    # Negative logic for and/or
    And = CriteriaBuilder.Or
    Or  = CriteriaBuilder.And


def expressionSignature(expr, mode):
    """Return a test that tests `expr` in truth `mode`"""
    # Default is to simply test the truth of the expression
    return Test(Truth(expr), Value(mode))    

def compileIn(expr, criterion, truth):
    """Return a signature or predicate (or None) for 'expr in criterion'"""
    try:
        iter(criterion)
    except TypeError:
        pass    # treat the in condition as a truth expression
    else:
        expr = Comparison(expr)
        values = [Test(expr, Value(v, truth)) for v in criterion]
        if truth:
            return Disjunction(values)
        else:
            return reduce(intersect, values)

when(compileIn, (object, type))
when(compileIn, (object, ClassType))
def compileInClass(expr, criterion, truth):
    return Test(IsInstance(expr), Class(criterion, truth))



class IndexedEngine(Engine, TreeBuilder):
    """A dispatching engine that builds trees using bitmap indexes"""

    def __init__(self, disp):
        self.signatures = []
        self.all_exprs = {}
        super(IndexedEngine, self).__init__(disp)

    def _add_method(self, signature, action):
        if signature not in self.registry:
            case_id = len(self.signatures)
            self.signatures.append(signature)
            requires = []
            exprs = self.all_exprs
            for _t, expr, criterion in tests_for(signature):
                if expr not in exprs:
                    exprs[expr] = 1
                    #if always_testable(expr):
                    #    Ordering(self, expr).requires([])
                #Ordering(self, expr).requires(requires)
                #requires.append(expr)
                BitmapIndex(self, expr).add_case(case_id, criterion)
        super(IndexedEngine, self)._add_method(signature, action)

    def _generate_code(self):
        smig = SMIGenerator(self.function)
        for expr in self.all_exprs: smig.maybe_cache(expr)
        memo = dict(
            [(expr, smig.action_id(self.to_expression(expr)))
                    for expr in self.all_exprs]
        )
        return smig.generate(self.build_root(memo)).func_code

    def _full_reset(self):
        # Replace the entire engine with a new one
        Dispatching(self.function).create_engine(self.__class__)





    synchronized()
    def seed_bits(self, expr, cases):
        return BitmapIndex(self, expr).seed_bits(cases)

    synchronized()
    def reseed(self, expr, criterion):
        return BitmapIndex(self, expr).reseed(criterion)

    # Make build() a synchronized method
    build = synchronized(TreeBuilder.build.im_func)

    def build_root(self, memo):
        return self.build(
            to_bits([len(self.signatures)])-1, frozenset(self.all_exprs), memo
        )

    def best_expr(self, cases, exprs):
        return super(IndexedEngine, self).best_expr(
            list(from_bits(cases)), exprs
        )

    def build_node(self, expr, cases, remaining_exprs, memo):
        return memo[expr], predicate_node_for(
            self, expr, cases, remaining_exprs, memo
        )

    def selectivity(self, expr, cases):
        return BitmapIndex(self, expr).selectivity(cases)

    def optimize(self, action):
        return action

    def to_expression(self, expr):
        return expr







    def build_leaf(self, cases, memo):
        action = self.rules.default_action
        signatures = self.signatures
        registry = self.registry
        for case_no in from_bits(cases):
            action = combine_actions(action, registry[signatures[case_no]])
        if action in memo:
            return memo[action]
        return memo.setdefault(action, (0, self.optimize(action)))


when(predicate_node_for, (IndexedEngine, Truth))
def truth_node(builder, expr, cases, remaining_exprs, memo):
    dont_cares, seedmap = builder.seed_bits(expr, cases)
    return (    # True/false tuple for Truth
        builder.build(seedmap[True, 0][0] | dont_cares, remaining_exprs, memo),
        builder.build(seedmap[False, 0][0] | dont_cares, remaining_exprs, memo)
    )

when(predicate_node_for, (IndexedEngine, Identity))
def identity_node(builder, expr, cases, remaining_exprs, memo):
    dont_cares, seedmap = builder.seed_bits(expr, cases)
    return dict(
        [(seed, builder.build(inc|dont_cares, remaining_exprs, memo))
            for seed, (inc, exc) in seedmap.iteritems()]
    )

when(predicate_node_for, (IndexedEngine, Comparison))
def range_node(builder, expr, cases, remaining_exprs, memo):
    dontcares, seedmap = builder.seed_bits(expr, cases)
    return split_ranges(
        dontcares, seedmap, lambda cases: builder.build(cases, remaining_exprs, memo)
    )

try: frozenset
except NameError: from core import frozenset





when(predicate_node_for, (IndexedEngine, IsInstance))
when(predicate_node_for, (IndexedEngine, IsSubclass))
def class_node(builder, expr, cases, remaining_exprs, memo):
    dontcares, seedmap = builder.seed_bits(expr, cases)
    cache = {}
    bitcache = {}
    defaults = (0, 0)

    def build_map(cls):
        bases = cls.__bases__
        try:
            inc, exc = seedmap[cls]
        except KeyError:
            if len(bases)>1:
                builder.reseed(expr, Class(cls))    # fix multiple inheritance
                inc, exc = seedmap.setdefault(
                    cls, builder.seed_bits(expr, cases)[1][cls]
                )
            else:
                inc = exc = 0
        if not bases and cls is not object:
            bases = (InstanceType,)
        for base in bases:
            try:
                i, e = bitcache[base]
            except KeyError:
                i, e = build_map(base)
            inc |= i
            exc |= e
        return bitcache.setdefault(cls, (inc, exc))

    def lookup_fn(cls):
        try:
            inc, exc = bitcache[cls]
        except KeyError:
            inc, exc = build_map(cls)
        return cache.setdefault(
            cls, builder.build(dontcares|(inc ^ (exc & inc)), remaining_exprs, memo)
        )
    return cache, lookup_fn

