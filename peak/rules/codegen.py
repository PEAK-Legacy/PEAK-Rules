from peak.util.assembler import *
from peak.util.symbols import Symbol
from ast_builder import build
try:
    set
except NameError:
    from sets import Set as set

__all__ = [
    'Getattr', 'GetSlice', 'BuildSlice', 'Dict', 'ExprBuilder', 'IfElse',
    'CSECode',
]

nodetype()
def Getattr(expr, attr, code=None):
    if code is None:
        return fold_args(Getattr, expr, attr)
    code(expr)
    return code.LOAD_ATTR(attr)

nodetype()
def GetSlice(expr, start=Pass, stop=Pass, code=None):
    if code is None:
        if expr is not Pass:
            return fold_args(GetSlice, expr, start, stop)
        return expr, start, stop
    code(expr)
    if start is not Pass:
        code(start)
        if stop is not Pass:
            return code(stop, Code.SLICE_3)
        return code.SLICE_1()
    elif stop is not Pass:
        code(stop)
        return code.SLICE_2()
    return code.SLICE_0()





nodetype()
def BuildSlice(start=Pass, stop=Pass, stride=Pass, code=None):
    if code is None:
        return fold_args(BuildSlice, start, stop, stride)
    if start is Pass: start = None
    if stop  is Pass: stop  = None
    code(start, stop, stride)
    if stride is not Pass:
        return code.BUILD_SLICE(3)
    return code.BUILD_SLICE(2)

nodetype()
def Dict(items, code=None):
    if code is None:
        return fold_args(Dict, tuple(map(tuple,items)))
    code.BUILD_MAP(0)
    for k,v in items:
        code.DUP_TOP()
        code(k, v)
        code.ROT_THREE()
        code.STORE_SUBSCR()

nodetype()
def IfElse(tval, cond, fval, code=None):
    if code is None:
        return fold_args(IfElse, tval, cond, fval)
    else_clause, end_if = Label(), Label()
    code(cond)
    if tval != cond:
        code(else_clause.JUMP_IF_FALSE, Code.POP_TOP, tval)
        if code.stack_size is not None:
            code(end_if.JUMP_FORWARD)
    elif fval != cond:
        code(end_if.JUMP_IF_TRUE)

    if fval !=cond:       
        return code(else_clause, Code.POP_TOP, fval, end_if)
    else:
        return code(else_clause, end_if)


def unaryOp(name, opcode):
    nodetype()
    def tmp(expr, code=None):
        if code is None:
            return fold_args(tmp, expr)
        return code(expr, opcode)
    tmp.__name__ = name
    return tmp

def binaryOp(name, opcode):
    nodetype()
    def tmp(left, right, code=None):
        if code is None:
            return fold_args(tmp, left, right)
        return code(left, right, opcode)
    tmp.__name__ = name
    return tmp

def listOp(name, opcode):
    nodetype()
    def tmp(items, code=None):
        if code is None:
            return fold_args(tmp, tuple(items))
        code(*items)
        return opcode(code, len(items))
    tmp.__name__ = name
    return tmp

def mkOps(optype, **ops):
    return dict([(name,optype(name, op)) for (name, op) in ops.items()])

def globalOps(optype, **ops):
    __all__.extend(ops)
    localOps(globals(), optype, **ops)

def localOps(ns, optype, **ops):
    ns.update(mkOps(optype, **ops))




globalOps(
    unaryOp,
    Not = Code.UNARY_NOT,
    Plus = Code.UNARY_POSITIVE,
    Minus = Code.UNARY_NEGATIVE,
    Repr = Code.UNARY_CONVERT,
    Invert = Code.UNARY_INVERT,
)

globalOps(
    binaryOp,
    Add = Code.BINARY_ADD,
    Sub = Code.BINARY_SUBTRACT,
    Mul = Code.BINARY_MULTIPLY,
    Div = Code.BINARY_DIVIDE,
    Mod = Code.BINARY_MODULO,
    FloorDiv = Code.BINARY_FLOOR_DIVIDE,
    Power = Code.BINARY_POWER,
    LeftShift = Code.BINARY_LSHIFT,
    RightShift = Code.BINARY_RSHIFT,
    Getitem = Code.BINARY_SUBSCR,
    Bitor = Code.BINARY_OR,
    Bitxor = Code.BINARY_XOR,
    Bitand = Code.BINARY_AND,
)

globalOps(
    listOp, Tuple = Code.BUILD_TUPLE, List = Code.BUILD_LIST
)












CACHE = Local('$CSECache')
SET_CACHE = lambda code: code.STORE_FAST(CACHE.name)

class CSETracker(Code):
    """Helper object that tracks common sub-expressions"""

    def __init__(self):
        super(CSETracker, self).__init__()
        self.cse_depends = {}

    def track(self, expr):
        self.track_stack = [None, 0]
        self.to_cache = []
        try:
            self(expr)
            return self.to_cache
        finally:
            del self.track_stack, self.to_cache

    def __call__(self, *args):
        scall = super(CSETracker, self).__call__
        ts = self.track_stack
        for ob in args:           
            ts[-1] += 1
            ts.append(ob)
            ts.append(0)
            try:
                scall(ob)
            finally:
                count = ts.pop()
                ts.pop()
            if count and callable(ob):
                # Only consider non-leaf callables for caching
                top = tuple(ts[-2:])
                if self.cse_depends.setdefault(ob, top) != top:
                    if ob not in self.to_cache:
                        self.to_cache.append(ob)




class CSECode(Code):
    """Code object with common sub-expression caching support"""

    def __init__(self):
        super(CSECode, self).__init__()
        self.expr_cache = {}
        self.tracker = CSETracker()
        
    def cache(self, expr):
        if not self.expr_cache:
            self.LOAD_CONST(None)
            self.STORE_FAST(CACHE.name)
        self.expr_cache.setdefault(
            expr, "%s #%d" % (expr, len(self.expr_cache)+1)
        )

    def maybe_cache(self, expr):
        map(self.cache, self.tracker.track(expr))

    def __call__(self, *args):
        scall = super(CSECode, self).__call__
        for ob in args:
            if callable(ob) and ob in self.expr_cache:
                key = self.expr_cache[ob]
                def calculate(code):
                    scall(ob, Code.DUP_TOP, CACHE, Const(key), Code.STORE_SUBSCR)
                cache = IfElse(
                    CACHE, CACHE, lambda c: scall({}, Code.DUP_TOP, SET_CACHE)
                )
                scall(
                    IfElse(
                        Getitem(CACHE, Const(key)),
                        Compare(Const(key), [('in', cache)]),
                        calculate
                    )
                )
            else:
                scall(ob)



class ExprBuilder:
    """Expression builder returning bytecode-able AST nodes"""

    def __init__(self,arguments,*namespaces):
        self.arguments = arguments
        self.namespaces = namespaces

    def Const(self,value):
        return Const(value)

    def Name(self,name):
        if name in self.arguments:
            return self.arguments[name]

        for ns in self.namespaces:
            if name in ns:
                return Const(ns[name])

        raise NameError(name)

    def Subscript(self, left, right):
        expr = build(self,left)
        key =  build(self,right)
        if isinstance(key, GetSlice):
            return GetSlice(expr, key.start, key.stop)
        return Getitem(expr, key)

    def Slice2(self, start, stop):
        start = start and build(self, start) or Pass
        stop  = stop  and build(self, stop ) or Pass
        return GetSlice(Pass, start, stop)

    def Slice3(self, start, stop, stride):
        start  = start  and build(self, start ) or Pass
        stop   = stop   and build(self, stop  ) or Pass
        stride = stride and build(self, stride) or Pass
        return BuildSlice(start, stop, stride)

    def Getattr(self, expr, attr):
        return Getattr(build(self,expr), attr)

    simplify_comparisons = False

    def Compare(self, expr, ops):
        return Compare(
            build(self, expr),
            [(op=='<>' and '!=' or op, build(self,arg)) for op, arg in ops]
        )

    def _unaryOp(name, nt):
        def method(self, expr):
            return nt(build(self,expr))
        return method

    localOps(locals(), _unaryOp,
        UnaryPlus  = Plus,
        UnaryMinus = Minus,
        Invert     = Invert,
        Backquote  = Repr,
        Not        = Not,
    )

    del _unaryOp

    def _mkBinOp(name, nt):
        def method(self, left, right):
            return nt(build(self,left), build(self,right))
        return method

    localOps(locals(), _mkBinOp,
        Add        = Add,
        Sub        = Sub,
        Mul        = Mul,
        Div        = Div,
        Mod        = Mod,
        FloorDiv   = FloorDiv,
        Power      = Power,
        LeftShift  = LeftShift,
        RightShift = RightShift,
    )
    del _mkBinOp

    def _multiOp(name, nt):
        def method(self, items):
            result = build(self,items[0])
            for item in items[1:]:
                result = nt(result, build(self,item))
            return result
        return method

    localOps(locals(), _multiOp,
        Bitor  = Bitor,
        Bitxor = Bitxor,
        Bitand = Bitand,
    )
    del _multiOp

    def _listOp(name, op):
        def method(self,items):
            return op(map(build.__get__(self), items))
        return method

    localOps(locals(), _listOp,
        And   = And,
        Or    = Or,
        Tuple = Tuple,
        List  = List,
    )

    def Dict(self, items):
        b = build.__get__(self)
        return Dict([(b(k),b(v)) for k,v in items])

    def CallFunc(self, func, args, kw, star_node, dstar_node):
        b = build.__get__(self)
        return Call(
            b(func), map(b,args), [(b(k),b(v)) for k,v in kw],
            star_node and b(star_node), dstar_node and b(dstar_node)
        )

    def IfElse(self, tval, cond, fval):
        return IfElse(build(self,tval), build(self,cond), build(self,fval))

