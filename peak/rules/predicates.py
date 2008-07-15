from peak.util.assembler import *
from core import *
from core import class_or_type_of
from criteria import *
from indexing import *
from codegen import SMIGenerator, ExprBuilder, Getitem, IfElse
from peak.util.decorators import decorate, synchronized, decorate_assignment
from types import InstanceType, ClassType
from ast_builder import build, parse_expr
import inspect

__all__ = [
    'IsInstance', 'IsSubclass', 'Truth', 'Identity', 'Comparison',
    'IndexedEngine', 'predicate_node_for', 'meta_function'
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

_unpack = lambda c: c.UNPACK_SEQUENCE(2)
subclass_check = TryExcept(
    Suite([
        Code.DUP_TOP, SMIGenerator.ARG, _unpack, Code.ROT_THREE,
        Code.POP_TOP, Code.BINARY_SUBSCR, Code.ROT_TWO, Code.POP_TOP
    ]), [(Const(KeyError), Suite([
        SMIGenerator.ARG, _unpack, Code.POP_TOP, Call(Code.ROT_TWO, (Pass,)),
    ]))]
)

nodetype()
def IsSubclass(expr, code=None):
    if code is None: return expr,
    code(expr, subclass_check)

identity_check = IfElse(
    Getitem(SMIGenerator.ARG, Code.ROT_TWO),
    Compare(Code.DUP_TOP, [('in', SMIGenerator.ARG)]),
    Suite([Code.POP_TOP, Getitem(SMIGenerator.ARG, None)])
)

nodetype()
def Identity(expr, code=None):
    if code is None: return expr,
    code(Call(Const(id), (expr,), fold=False), identity_check)

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


class ExprBuilder(ExprBuilder):
    """Extended expression builder with support for meta-functions"""

    def Backquote(self, expr):
        raise SyntaxError("backquotes are not allowed in predicates")

meta_functions = {}

def meta_function(*stub, **parsers):
    """Declare a meta-function and its argument parsers"""
    stub, = stub
    def callback(frame, name, func, old_locals):
        for name in inspect.getargs(func.func_code)[0]:
            if not isinstance(name, basestring):
                raise TypeError(
                    "Meta-functions cannot have packed-tuple arguments"
                )
        meta_functions[stub] = func, parsers, inspect.getargspec(func)
        return func
    return decorate_assignment(callback)





















class CriteriaBuilder:

    simplify_comparisons = True
    parse = ExprBuilder.parse.im_func
    def __init__(self, arguments, *namespaces):
        self.expr_builder = ExprBuilder(arguments, *namespaces)

    def mkOp(name):
        op = getattr(ExprBuilder,name)
        def method(self, *args):
            return expressionSignature(op(self.expr_builder, *args))
        return method

    for opname in dir(ExprBuilder):
        if opname[0].isalpha() and opname[0]==opname[0].upper():
            locals()[opname] = mkOp(opname)

    def Not(self, expr):
        return negate(self.build_with(expr))

    def build_with(self, expr, ):
        self.expr_builder.push()
        try:
            return build(self, expr)
        finally:
            self.expr_builder.pop()

    _mirror_ops = {
        '>': '<', '>=': '<=', '=>':'<=',
        '<': '>', '<=': '>=', '=<':'>=',
        '<>': '<>', '!=': '<>', '==':'==',
        'is': 'is', 'is not': 'is not'
    }








    def Compare(self, initExpr, ((op,other),)):
        old_op = [op, '!='][op=='<>']
        left = initExpr = build(self.expr_builder, initExpr)
        right = other = build(self.expr_builder, other)
        if isinstance(left,Const) and op in self._mirror_ops:
            left, right, op = right, left, self._mirror_ops[op]

        if isinstance(right,Const):
            if op=='in' or op=='not in':
                cond = compileIn(left, right.value)
                if cond is not None:
                    return maybe_invert(cond, op=='in')
            elif op=='is' or op=='is not':
                return maybe_invert(compileIs(left, right.value), op=='is')
            else:
                return Test(Comparison(left), Inequality(op, right.value))

        # Both sides involve variables or an un-optimizable constant,
        #  so it's a generic boolean criterion  :(
        return expressionSignature(
            Compare(initExpr, ((old_op, other),))
        )

    def And(self, items):
        return reduce(intersect, [build(self,expr) for expr in items], True)

    def Or(self, items):
        return OrElse(map(self.build_with, items))

    def CallFunc(self, func, args, kw, star_node, dstar_node):
        b = build.__get__(self.expr_builder)
        target = b(func)
        if isinstance(target, Const) and target.value in meta_functions:
            return self.apply_meta(
                meta_functions[target.value], args, kw, star_node, dstar_node
            )
        return expressionSignature(Call(
            target, map(b,args), [(b(k),b(v)) for k,v in kw],
            star_node and b(star_node), dstar_node and b(dstar_node)
        ))

    def apply_meta(self,
        (func, parsers, (argnames, varargs, varkw, defaults)), args, kw, star, dstar
    ):
        # NB: tuple-args not allowed!
        def parse(arg, node):
            if not node:
                return None
            return parsers.get(arg, build)(self.expr_builder, node)

        data = {}
        extra = []
        offset = 0
        for name in argnames:
            if name=='__builder__': data[name] = self.expr_builder
            elif name=='__star__':  data[name] = parse(name, star)
            elif name=='__dstar__': data[name] = parse(name, dstar)
            else:
                break
            offset += 1

        for k, v in zip(argnames[offset:], args):
            data[k] = parse(k, v)

        varargpos = len(argnames)-offset
        if len(args)> varargpos:
            if not varargs:
                raise TypeError("Too many arguments for %r" % (func,))
            extra.extend([parse(varargs, node) for node in args[varargpos:]])

        for k,v in kw:
            k = build(self.expr_builder, k)
            assert type(k) is Const and isinstance(k.value, basestring)
            k = k.value
            if k in data:
                raise TypeError("Duplicate keyword %s for %r" % (k,func))

            if varkw and k not in argnames and k not in parsers:
                data[k] = parse(varkw,  v)
            else:
                data[k] = parse(k, v)

        if star and '__star__' not in data:
            raise TypeError("%r does not support parsing *args" % (func,))

        if dstar and '__dstar__' not in data:
            raise TypeError("%r does not support parsing **kw" % (func,))

        if defaults:
            for k,v in zip(argnames[-len(defaults):], defaults):
                data.setdefault(k, v)

        try:
            args = map(data.pop, argnames)+extra
        except KeyError, e:
            raise TypeError(
                "Missing positional argument %s for %r"%(e.args[0], func)
            )
        return func(*args, **data)
























def compileIs(expr, criterion):
    """Return a signature or predicate for 'expr is criterion'"""
    #if criterion is None:     # XXX this should be smarter
    #    return Test(IsInstance(expr), istype(NoneType))
    #else:
    return Test(Identity(expr), IsObject(criterion))

def maybe_invert(cond, truth):
    if not truth: return negate(cond)
    return cond

def expressionSignature(expr):
    """Return a test that tests `expr` in truth `mode`"""
    # Default is to simply test the truth of the expression
    return Test(Truth(expr), Value(True))

when(expressionSignature, (Const,))(lambda expr: bool(expr.value))

def compileIn(expr, criterion):
    """Return a signature or predicate (or None) for 'expr in criterion'"""
    try:
        iter(criterion)
    except TypeError:
        pass    # treat the in condition as a truth expression
    else:
        expr = Comparison(expr)
        return Test(expr, Disjunction([Value(v) for v in criterion]))

when(compileIn, (object, type))
when(compileIn, (object, ClassType))
def compileInClass(expr, criterion):
    return Test(IsInstance(expr), Class(criterion))

when(compileIn, (object, istype))
def compileInIsType(expr, criterion):
    return Test(IsInstance(expr), criterion)





class IndexedEngine(Engine, TreeBuilder):
    """A dispatching engine that builds trees using bitmap indexes"""

    def __init__(self, disp):
        self.signatures = []
        self.all_exprs = {}
        super(IndexedEngine, self).__init__(disp)
        self.arguments = dict([(arg,Local(arg)) for arg in self.argnames])

    def _add_method(self, signature, rule):
        signature = Signature(tests_for(signature, self))
        if signature not in self.registry:
            case_id = len(self.signatures)
            self.signatures.append(signature)
            requires = []
            exprs = self.all_exprs
            for _t, expr, criterion in tests_for(signature, self):
                if expr not in exprs:
                    exprs[expr] = 1
                    if always_testable(expr):
                        Ordering(self, expr).requires([])
                Ordering(self, expr).requires(requires)
                requires.append(expr)
                BitmapIndex(self, expr).add_case(case_id, criterion)
        return super(IndexedEngine, self)._add_method(signature, rule)

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

when(bitmap_index_type,  (IndexedEngine, Truth))(lambda en,ex:TruthIndex)
when(predicate_node_for, (IndexedEngine, Truth))
def truth_node(builder, expr, cases, remaining_exprs, memo):
    dont_cares, seedmap = builder.seed_bits(expr, cases)
    return (    # True/false tuple for Truth
        builder.build(seedmap[True][0] | dont_cares, remaining_exprs, memo),
        builder.build(seedmap[False][0] | dont_cares, remaining_exprs, memo)
    )

when(bitmap_index_type,  (IndexedEngine, Identity))(lambda en,ex:PointerIndex)
when(predicate_node_for, (IndexedEngine, Identity))
def identity_node(builder, expr, cases, remaining_exprs, memo):
    dont_cares, seedmap = builder.seed_bits(expr, cases)
    return dict(
        [(seed, builder.build(inc|dont_cares, remaining_exprs, memo))
            for seed, (inc, exc) in seedmap.iteritems()]
    )

when(bitmap_index_type,  (IndexedEngine, Comparison))(lambda en,ex:RangeIndex)
when(predicate_node_for, (IndexedEngine, Comparison))
def range_node(builder, expr, cases, remaining_exprs, memo):
    dontcares, seedmap = builder.seed_bits(expr, cases)
    return split_ranges(
        dontcares, seedmap, lambda cases: builder.build(cases, remaining_exprs, memo)
    )

try: frozenset
except NameError: from core import frozenset



when(bitmap_index_type,  (IndexedEngine, IsInstance))(lambda en,ex:TypeIndex)
when(bitmap_index_type,  (IndexedEngine, IsSubclass))(lambda en,ex:TypeIndex)

when(predicate_node_for, (IndexedEngine, IsInstance))
when(predicate_node_for, (IndexedEngine, IsSubclass))
def class_node(builder, expr, cases, remaining_exprs, memo):
    dontcares, seedmap = builder.seed_bits(expr, cases)
    cache = {}
    def lookup_fn(cls):
        try:
            inc, exc = seedmap[cls]
        except KeyError:
            builder.reseed(expr, Class(cls))
            seedmap.update(builder.seed_bits(expr, cases)[1])
            inc, exc = seedmap[cls]
        cbits = dontcares | inc
        cbits ^= (exc & cbits)
        return cache.setdefault(cls, builder.build(cbits,remaining_exprs,memo))

    return cache, lookup_fn


abstract()
def type_to_test(typ, expr, engine):
    """Convert `typ` to a ``Test()`` of `expr` for `engine`"""

when(type_to_test, (type,))
when(type_to_test, (ClassType,))
def std_type_to_test(typ, expr, engine):
    return Test(IsInstance(expr), Class(typ))

when(type_to_test, (istype,))
def istype_to_test(typ, expr, engine):
    return Test(IsInstance(expr), typ)







when(tests_for, (istype(tuple), Engine))
def tests_for_tuple(ob, engine):
    for cls, arg in zip(ob, engine.argnames):
        yield type_to_test(cls, Local(arg), engine)

def always_testable(expr):
    """Is `expr` safe to evaluate in any order?"""
    return False

when(always_testable, (IsInstance,))
#when(always_testable, (IsSubclass,))   XXX might not be a class!
when(always_testable, (Identity,))
when(always_testable, (Truth,))
when(always_testable, (Comparison,))
def testable_criterion(expr):
    return always_testable(expr.expr)

when(always_testable, (Local,))(lambda expr:True)
when(always_testable, (Const,))(lambda expr:True)


when(parse_rule, (IndexedEngine, basestring))
def _parse_string(engine, predicate, ctx, cls):
    b = CriteriaBuilder(engine.arguments, ctx.localdict, ctx.globaldict, __builtins__)
    expr = parse_expr(predicate, b)
    if cls is not None and engine.argnames:
        cls = type_to_test(cls, engine.arguments[engine.argnames[0]], engine)
        expr = intersect(cls, expr)
    return Rule(ctx.body, expr, ctx.actiontype, ctx.sequence)












# === As of this point, it should be possible to compile expressions!
#
meta_function(isinstance)(
    lambda __star__, __dstar__, *args, **kw:
        compileIsXCall(isinstance, IsInstance, args, kw, __star__, __dstar__)
)
meta_function(issubclass)(
    lambda __star__, __dstar__, *args, **kw:
        compileIsXCall(issubclass, IsSubclass, args, kw, __star__, __dstar__)
)
def compileIsXCall(func, test, args, kw, star, dstar):    
    if (
        kw or len(args)!=2 or not isinstance(args[1], Const)
        or isinstance(args[0], Const) or star is not None or dstar is not None
    ):
        return expressionSignature(
            Call(Const(func), args, tuple(kw.items()), star, dstar)
        )
    expr, seq = args
    return Test(test(expr), Disjunction(map(Class, _yield_tuples(seq.value))))

def _yield_tuples(ob):
    if type(ob) is tuple:
        for i1 in ob:
            for i2 in _yield_tuples(i1):
                yield i2
    else:
        yield ob

when(compileIs,
    # matches 'type(x) is y'
    "expr in Call and expr.func in Const and (expr.func.value is type)"
    " and len(expr.args)==1"
)
def compileTypeIsX(expr, criterion):
    return Test(IsInstance(expr.args[0]), istype(criterion))





