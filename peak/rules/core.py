__all__ = [
    'Rule', 'RuleSet', 'Dispatching', 'Engine', 'rules_for',
    'Method', 'Around', 'Before', 'After', 'MethodList',
    'DispatchError', 'AmbiguousMethods', 'NoApplicableMethods',
    'abstract', 'when', 'before', 'after', 'around', 'istype', 'parse_rule',
    'implies', 'dominant_signatures', 'combine_actions', 'overrides',
    'always_overrides', 'merge_by_default', 'intersect', 'disjuncts', 'negate'
]
from peak.util.decorators import decorate_assignment, decorate, struct, synchronized, frameinfo, decorate_class
from peak.util.assembler import Code, Const, Call, Local, Getattr, TryExcept, Suite, with_name
from peak.util.addons import AddOn
import inspect, new, itertools, operator
try:
    set = set
    frozenset = frozenset
except NameError:
    from sets import Set as set
    from sets import ImmutableSet
    class frozenset(ImmutableSet):
        """Kludge to fix the abomination that is ImmutableSet.__init__"""
        def __new__(cls, iterable=None):
            self = ImmutableSet.__new__(cls, iterable)
            ImmutableSet.__init__(self, iterable)
            return self
        def __init__(self, iterable=None):
            pass    # all immutable initialization should be done by __new__!

try:
    sorted = sorted
except NameError:
    def sorted(seq,key=None):
        if key:
            d = [(key(v),v) for v in seq]
        else:
            d = list(seq)
        d.sort()
        if key:
            return [v[1] for v in d]
        return d
empty = frozenset()

next_sequence = itertools.count().next

struct()
def Rule(body, predicate=(), actiontype=None, sequence=None):
    if sequence is None:
        sequence = next_sequence()
    return body, predicate, actiontype, sequence

struct()
def ParseContext(
    body, actiontype=None, localdict=(), globaldict=(), sequence=None
):
    """Hold information needed to parse a predicate"""
    if sequence is None:
        sequence = next_sequence()
    return body, actiontype, dict(localdict), dict(globaldict), sequence

def disjuncts(ob):
    """Return a *list* of the logical disjunctions of `ob`"""
    # False == no condition is sufficient == no disjuncts
    if ob is False: return []
    return [ob]

def parse_rule(engine, predicate, context, cls):
    """Hook for pre-processing predicates, e.g. parsing string expressions"""
    if cls is not None and type(predicate) is tuple:
        predicate = (cls,) + predicate        
    return Rule(context.body, predicate, context.actiontype, context.sequence)

def clone_function(f):
    return new.function(
      f.func_code, f.func_globals, f.func_name, f.func_defaults, f.func_closure
    )








[struct()]
def istype(type, match=True):
    return type, match

def type_key(arg):
    if isinstance(arg, (type, ClassType)):
        return arg
    elif type(arg) is istype and arg.match:
        return arg.type

def type_keys(sig):
    if type(sig) is not tuple:
        return
    key = tuple(map(type_key, sig))
    if None not in key:
        yield key

def YES(s1,s2): return True
def NO(s1,s2):  return False

def always_overrides(a, b):
    """`a` instances always override `b`s; `b` instances never override `a`s"""
    a,b = istype(a), istype(b)
    when(overrides, (a, b))(YES)
    when(overrides, (b, a))(NO)
    pairs = {}; to_add = []
    for rule in rules_for(overrides):
        sig = rule.predicate
        if type(sig) is not tuple or len(sig)!=2 or rule.body is not YES:
            continue
        pairs[sig]=1
        if sig[0]==b: to_add.append((a, sig[1]))
        if sig[1]==a: to_add.append((sig[0], b))
    for (p1,p2) in to_add:
        if (p1,p2) not in pairs:
            always_overrides(p1.type, p2.type)

def merge_by_default(t):
    """instances of `t` never imply other instances of `t`"""
    when(overrides, (t, t))(NO)

class Method(object):
    """A simple method w/optional chaining"""

    def __init__(self, body, signature=(), precedence=0, tail=None):
        self.body = body
        self.signature = signature
        self.precedence = precedence
        self.tail = tail
        self.can_tail = False
        try:
            args = inspect.getargspec(body)[0]
        except TypeError:
            pass
        else:
            if args and args[0]=='next_method':
                if getattr(body, 'im_self', None) is None:
                    self.can_tail = True

    decorate(classmethod)
    def make(cls, body, signature=(), precedence=0):
        return cls(body, signature, precedence)

    def __repr__(self):
        data = (self.body, self.signature, self.precedence, self.tail)
        return self.__class__.__name__+repr(data)

    def __call__(self, *args, **kw):
        if self.can_tail:
            return self.body(self.tail, *args, **kw)
        return self.body(*args, **kw)

    def override(self, other):
        if not self.can_tail:
            return self
        return self.tail_with(combine_actions(self.tail, other))

    def tail_with(self, tail):
        return self.__class__(self.body, self.signature, self.precedence, tail)



    def merge(self, other):
        #if self.__class__ is other.__class__ and self.body is other.body:
        #    XXX precedence should also match; need to merge signatures
        #    return self.__class__(
        #        self.body, ???, ???, combine_actions(self.tail, other.tail)
        #    )
        return AmbiguousMethods([self,other])

    decorate(classmethod)
    def make_decorator(cls, name, doc=None):
        if doc is None:
            doc = "Extend a generic function with a method of type ``%s``" \
                  % cls.__name__
        if cls is Method:
            maker = None   # allow gf's to use something else instead of Method
        else:
            maker = cls.make
        def decorate(f, pred=(), depth=2, frame=None):
            def callback(frame, name, func, old_locals):
                assert f is not func    # XXX
                rules = rules_for(f)
                engine = Dispatching(f).engine
                kind, module, locals_, globals_ = frameinfo(frame)
                context = ParseContext(func, maker, locals_, globals_)
                def register_for_class(cls):
                    rules.add(parse_rule(engine, pred, context, cls))
                    return cls

                if kind=='class':
                    # 'when()' in class body; defer adding the method
                    decorate_class(register_for_class, frame=frame)
                else:
                    register_for_class(None)
                if old_locals.get(name) in (f, rules):
                    return f    # prevent overwriting if name is the same
                return func
            return decorate_assignment(callback, depth, frame)
        decorate = with_name(decorate, name)
        decorate.__doc__ = doc
        return decorate

when = Method.make_decorator(
    "when", "Extend a generic function with a new action"
)

class DispatchError(Exception):
    """A dispatch error has occurred"""

    def __call__(self,*args,**kw):
        raise self.__class__(*self.args+(args,kw))  # XXX

    def __repr__(self):
        # This method is needed so doctests for 2.3/2.4 match 2.5
        return self.__class__.__name__+repr(self.args)

class NoApplicableMethods(DispatchError):
    """No applicable action has been defined for the given arguments"""

    def merge(self, other):
        return AmbiguousMethods([self,other])

class AmbiguousMethods(DispatchError):
    """More than one choice of action is possible"""

    def __init__(self, methods, *args):
        DispatchError.__init__(self, methods, *args)
        mine = self.methods = []
        for m in methods:
            if isinstance(m, AmbiguousMethods):
                mine.extend(m.methods)
            else:
                mine.append(m)

    def merge(self, other):
        return AmbiguousMethods(self.methods+[other])

    def override(self, other):
        return self
    def __repr__(self): return "AmbiguousMethods(%s)" % self.methods



class RuleSet(object):
    """An observable, stably-ordered collection of rules"""
    default_action = NoApplicableMethods()
    default_actiontype = Method
    counter = 0

    def __init__(self, lock=None):
        self.rules = []
        self.actiondefs = {}
        self.listeners = []
        if lock is not None:
            self.__lock__ = lock

    synchronized()
    def add(self, rule):
        actiondefs = frozenset(self._actions_for(rule))
        self.rules.append( rule )
        self.actiondefs[rule] = actiondefs
        self._notify(added=actiondefs)

    synchronized()
    def remove(self, rule):
        actiondefs = self.actiondefs.pop(rule)
        self.rules.remove(rule)
        self._notify(removed=actiondefs)

    synchronized()
    def clear(self):
        actiondefs = frozenset(self)
        del self.rules[:]; self.actiondefs.clear()
        self._notify(removed=actiondefs)
    #def changed(self, rule):
    #    sequence, actions = self.actions[rule]
    #    new_actions = frozenset(self._actions_for(rule, sequence))
    #    self.actions[rule] = sequence, new_actions
    #    self.notify(new_actions-actions, actions-new_actions)

    def _notify(self, added=empty, removed=empty):
        for listener in self.listeners[:]:  # must be re-entrant
            listener.actions_changed(added, removed)
            
    synchronized()
    def __iter__(self):
        ad = self.actiondefs
        return iter([a for rule in self.rules for a in ad[rule]])

    def _actions_for(self, (na, body, predicate, actiontype, seq)):
        actiontype = actiontype or self.default_actiontype
        for signature in disjuncts(predicate):
            yield Rule(body, signature, actiontype, seq)

    synchronized()
    def subscribe(self, listener):
        self.listeners.append(listener)
        if self.rules:
            listener.actions_changed(frozenset(self), empty)

    synchronized()
    def unsubscribe(self, listener):
        self.listeners.remove(listener)


def rules_for(f):
    """Return the initialized ruleset for a generic function"""
    if not Dispatching.exists_for(f):
        d = Dispatching(f)
        d.rules.add(Rule(clone_function(f)))
    return Dispatching(f).rules

def abstract(func=None):
    """Declare a function to be abstract"""
    if func is None:
        return decorate_assignment(
            lambda f,n,func,old: Dispatching(func).as_abstract()
        )
    else:
        return Dispatching(func).as_abstract()





class Dispatching(AddOn):
    """Hold the dispatching attributes of a generic function"""
    engine = None
    def __init__(self, func):
        self.function = func
        self._regen   = self._regen_code()
        self.rules    = RuleSet(self.get_lock())
        self.backup   = None
        self.create_engine(TypeEngine)

    synchronized()
    def get_lock(self):
        return self.__lock__

    def create_engine(self, engine_type):
        """Create a new engine of `engine_type`, unsubscribing old"""
        if self.engine is not None and self.engine in self.rules.listeners:
            self.rules.unsubscribe(self.engine)
        self.engine = engine_type(self)
        return self.engine

    synchronized()
    def request_regeneration(self):
        """Ensure code regeneration occurs on next call of the function"""
        if self.backup is None:
            self.backup = self.function.func_code
            self.function.func_code = self._regen

    def _regen_code(self):
        c = Code.from_function(self.function, copy_lineno=True)
        c.return_(
            call_thru(
                self.function,
                Call(Getattr(
                    Call(Const(Dispatching), (Const(self.function),), fold=False),
                    '_regenerate'
                ))
            )
        )
        return c.code()

    synchronized()
    def as_abstract(self):
        for action in self.rules:
            raise AssertionError("Can't make abstract: rules already exist")

        c = Code.from_function(self.function, copy_lineno=True)
        c.return_(call_thru(self.function, Const(self.rules.default_action)))

        if self.backup is None:
            self.function.func_code = c.code()
        else:
            self.backup = c.code()
        return self.function

    synchronized()
    def _regenerate(self):
        func = self.function
        assert self.backup is not None
        func.func_code = self.backup    # ensure re-entrant calls work

        try:
            # try to replace the code with new code
            func.func_code = self.engine._generate_code()
        except:
            # failure: we'll try to regen next time we're called
            func.func_code = self._regen
            raise
        else:
            # success!  get rid of the old backup code and return the function
            self.backup = None
            return func










class Engine(object):
    """Abstract base for dispatching engines"""

    reset_on_remove = True

    def __init__(self, disp):
        self.function = disp.function
        self.registry = {}
        self.rules = disp.rules
        self.__lock__ = disp.get_lock()
        self.argnames = list(
            flatten(filter(None, inspect.getargspec(self.function)[:3]))
        )
        self.rules.subscribe(self)

    synchronized()
    def actions_changed(self, added, removed):
        if removed and self.reset_on_remove:
            return self._full_reset()
        for rule in removed:
            self._remove_method(rule.predicate, rule)
        for rule in added:
            self._add_method(rule.predicate, rule)
        if added or removed:
            self._changed()

    def _changed(self):
        """Some change to the rules has occurred"""
        Dispatching(self.function).request_regeneration()

    def _full_reset(self):
        """Regenerate any code, caches, indexes, etc."""
        self.registry.clear()
        self.actions_changed(self.rules, ())
        Dispatching(self.function).request_regeneration()






    def _add_method(self, signature, rule):
        """Add a case for the given signature and rule"""
        registry = self.registry
        action = rule.actiontype(rule.body, signature, rule.sequence)
        if signature in registry:
            registry[signature] = combine_actions(registry[signature], action)
        else:
            registry[signature] = action
        return action

    def _remove_method(self, signature, rule):
        """Remove the case for the given signature and rule"""
        raise NotImplementedError

    def _generate_code(self):
        """Return a code object for the current state of the function"""
        raise NotImplementedError
























class TypeEngine(Engine):
    """Simple type-based dispatching"""

    cache = None

    def __init__(self, disp):
        self.static_cache = {}
        super(TypeEngine, self).__init__(disp)

    def _changed(self):
        if self.cache != self.static_cache:
            Dispatching(self.function).request_regeneration()

    def _bootstrap(self):
        """Bootstrap a self-referential generic function"""
        cache = self.static_cache
        for sig, act in self.registry.items():
            for key in type_keys(sig):
                cache[key] = act
        self._changed()

    def _add_method(self, signature, rule):
        action = super(TypeEngine, self)._add_method(signature, rule)
        cache = self.static_cache
        for key in cache.keys():
            if key!=signature and implies(key, signature):
                cache[key] = combine_actions(cache[key], action)
        return action













    def _generate_code(self):
        self.cache = cache = self.static_cache.copy()
        def callback(*args, **kw):
            types = tuple([getattr(arg,'__class__',type(arg)) for arg in args])
            self.__lock__.acquire()
            try:
                action = self.rules.default_action
                for sig in self.registry:
                    if sig==types or implies(types, sig):
                        action = combine_actions(action, self.registry[sig])
                f = cache[types] = action
            finally:
                self.__lock__.release()
            return f(*args, **kw)

        c = Code.from_function(self.function, copy_lineno=True)
        types = [class_or_type_of(Local(name))
                    for name in flatten(inspect.getargspec(self.function)[0])]
        target = Call(Const(cache.get), (tuple(types), Const(callback)))
        c.return_(call_thru(self.function, target))
        return c.code()

def flatten(v):
    if isinstance(v,basestring): yield v; return
    for i in v:
        for ii in flatten(i): yield ii

def gen_arg(v):
    if isinstance(v,basestring): return Local(v)
    if isinstance(v,list): return tuple(map(gen_arg,v))

def call_thru(sigfunc, target):
    args, star, dstar, defaults = inspect.getargspec(sigfunc)
    return Call(target, map(gen_arg,args), (), gen_arg(star), gen_arg(dstar), fold=False)

def class_or_type_of(expr):
    return Suite([expr, TryExcept(
        Suite([Getattr(Code.DUP_TOP, '__class__'), Code.ROT_TWO, Code.POP_TOP]),
        [(Const(AttributeError), Call(Const(type), (Code.ROT_TWO,)))]
    )])

def overrides(a1, a2):
    return False

def combine_actions(a1,a2):
    """Return a new action for the combination of a1 and a2"""
    if a1 is None:
        return a2
    elif a2 is None:
        return a1
    elif overrides(a1,a2):
        if not overrides(a2,a1):
            return a1.override(a2)
    elif overrides(a2,a1):
        return a2.override(a1)
    return a1.merge(a2)

def implies(s1,s2):
    """Is s2 always true if s1 is true?"""
    return s1==s2

when(implies, (istype(tuple), istype(tuple)))
def tuple_implies(s1,s2):
    if len(s2)>len(s1):
        return False    # shorter tuple can't imply longer tuple
    for t1,t2 in zip(s1,s2):
        if not implies(t1,t2):
            return False
    else:
        return True

from types import ClassType, InstanceType
when(implies, (type,      type)     )(issubclass)
when(implies, (ClassType, ClassType))(issubclass)
when(implies, (type,      ClassType))(issubclass)
when(implies, (type,      istype)   )(lambda s1,s2: s2.match==(s1 is s2.type))
when(implies, (istype,    istype)   )(lambda s1,s2:
    s1==s2 or (s1.type is not s2.type and s1.match and not s2.match))
when(implies, (istype,type))(lambda s1,s2: s1.match and issubclass(s1.type,s2))



when(implies, (ClassType, type))
def classic_implies_new(s1, s2):
    # A classic class only implies a new-style one if it's ``object``
    # or ``InstanceType``; this is an exception to the general rule that
    # isinstance(X,Y) implies issubclass(X.__class__,Y)
    return s2 is object or s2 is InstanceType

# ok, implies() is now ready to rumble
Dispatching(implies).engine._bootstrap()

when(overrides, (Method,Method))
def method_overrides(a1, a2):
    if a1.__class__ is a2.__class__:
        return implies(a1.signature, a2.signature)
    raise TypeError("Incompatible action types", a1, a2)

class Around(Method):
    """'Around' Method (takes precedence over regular methods)"""

around = Around.make_decorator('around')

always_overrides(Around, Method)



















class MethodList(Method):
    """A list of related methods"""
    def __init__(self, items=(), tail=None):
        self.items = list(items)
        self.tail = tail
        self.can_tail = True

    _sorted_items = None

    decorate(classmethod)
    def make(cls, body, signature=(), precedence=0):
        return cls( [(signature, precedence, body)] )

    def __repr__(self):
        data = self.items, self.tail
        return self.__class__.__name__+repr(data)

    def tail_with(self, tail):
        return self.__class__(self.items, tail)

    def merge(self, other):
        if other.__class__ is not self.__class__:
            raise TypeError("Incompatible action types for merge", self, other)
        return self.__class__(
            self.items+other.items, combine_actions(self.tail, other.tail)
        )















    def sorted(self):
        if self._sorted_items is not None:
            return self._sorted_items

        self.items.sort(lambda a,b: cmp(a[1],b[1]))
        rest = [(s,b) for (s,p,b) in self.items]

        self._sorted_items = items = []
        seen = set()
        while rest:
            best = dominant_signatures(rest)
            map(rest.remove, best)
            for s,b in best:
                if b not in seen:
                    seen.add(b)
                    items.append((s,b))
        return items

merge_by_default(MethodList)

abstract()
def intersect(c1, c2):
    """Return the logical intersection of two conditions"""

around(intersect, (object, object))
def intersect_if_implies(next_method, c1, c2):
    if implies(c1,c2):      return c1
    elif implies(c2, c1):   return c2
    return next_method(c1, c2)

# These are needed for boolean intersects to work correctly
when(implies, (bool, bool))(lambda c1, c2: c2 or not c1)
when(implies, (bool, object))(lambda c1, c2: not c1)
when(implies, (object, bool))(lambda c1, c2: c2)







class Before(MethodList):
    """Method(s) to be called before the primary method(s)"""

    def __call__(self, *args, **kw):
        for sig, body in self.sorted():
            body(*args, **kw)
        return self.tail(*args, **kw)

before = Before.make_decorator('before')
merge_by_default(Before)

class After(MethodList):
    """Method(s) to be called after the primary method(s)"""

    def sorted(self):
        # Reverse the sorting for after methods
        if self._sorted_items is not None:
            return self._sorted_items
        items = super(After,self).sorted()
        items.reverse()
        return items

    def __call__(self, *args, **kw):
        retval = self.tail(*args, **kw)
        for sig, body in self.sorted():
            body(*args, **kw)
        return retval

after  = After.make_decorator('after')
merge_by_default(After)

always_overrides(Around, Before)
always_overrides(Before, After)
always_overrides(After, Method)

merge_by_default(DispatchError)
when(overrides, (Method, NoApplicableMethods))(YES)
when(overrides, (NoApplicableMethods, Method))(NO)

Dispatching(overrides).engine._bootstrap()

when(overrides, (AmbiguousMethods, Method))
def ambiguous_overrides(a1, a2):
    for m in a1.methods:
        if overrides(m, a2):
            # if any ambiguous method overrides a2, we can toss it
            return True
    return False

when(overrides, (Method, AmbiguousMethods))
def override_ambiguous(a1, a2):
    for m in a2.methods:
        if not overrides(a1, m):
            return False
    return True     # can only override if it overrides all the ambiguity

# needed to disambiguate the above two methods if combining a pair of AM's:
merge_by_default(AmbiguousMethods)


when(parse_rule, (TypeEngine, basestring))
def parse_string_rule_by_upgrade(engine, predicate, context, cls):
    """Upgrade to predicate dispatch engine and do the parse"""
    from peak.rules.predicates import IndexedEngine
    return parse_rule(
        Dispatching(engine.function).create_engine(IndexedEngine),
        predicate, context, cls
    )

when(rules_for, type(After.sorted))(lambda f: rules_for(f.im_func))


abstract()
def negate(c):
    """Return the logical negation of criterion `c`"""

when(negate, (bool,)  )(operator.not_)
when(negate, (istype,))(lambda c: istype(c.type, not c.match))




def dominant_signatures(cases):
    """Return the most-specific ``(signature,body)`` pairs from `cases`

    `cases` is a sequence of ``(signature,body)`` pairs.  This routine checks
    the ``implies()`` relationships between pairs of signatures, and then
    returns a list of ``(signature,method)`` pairs such that no signature
    remaining in the original list implies a signature in the new list.
    The relative order of cases in the new list is preserved.
    """

    if len(cases)==1:
        # Shortcut for common case
        return list(cases)

    best, rest = list(cases[:1]), list(cases[1:])

    for new_sig, new_meth in rest:

        for old_sig, old_meth in best[:]:   # copy so we can modify inplace

            new_implies_old = implies(new_sig, old_sig)
            old_implies_new = implies(old_sig, new_sig)

            if new_implies_old:

                if not old_implies_new:
                    # better, remove the old one
                    best.remove((old_sig, old_meth))

            elif old_implies_new:
                # worse, skip adding the new one
                break
        else:
            # new_sig has passed the gauntlet, as it has not been implied
            # by any of the current "best" items
            best.append((new_sig,new_meth))

    return best



