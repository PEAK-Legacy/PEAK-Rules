__all__ = [
    'Rule', 'RuleSet', 'predicate_signatures', 'rules_for',
    'Method', 'Around', 'Before', 'After', 'MethodList',
    'DispatchError', 'AmbiguousMethods', 'NoApplicableMethods',
    'abstract', 'when', 'before', 'after', 'around',
    'implies', 'dominant_signatures', 'combine_actions',
    'always_overrides', 'merge_by_default',
]

from peak.util.decorators import decorate_assignment, decorate, struct
from peak.util.assembler import Code, Const, Call, Local
import inspect, new

try:
    set, frozenset
except NameError:
    from sets import Set as set
    from sets import ImmutableSet as frozenset

empty = frozenset()

struct()
def Rule(body, predicate=(), actiontype=None):
    return body, predicate, actiontype

struct()
def ActionDef(actiontype, body, signature, sequence):
    return actiontype, body, signature, sequence

def predicate_signatures(predicate):
    yield predicate # XXX

_core_rules = None      # the core rules aren't loaded yet

def parse_rule(ruleset, body, predicate, actiontype, localdict, globaldict):
    """Hook for pre-processing predicates, e.g. parsing string expressions"""
    return Rule(body, predicate, actiontype)




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

        def decorate(f, pred=()):
            rules = rules_for(f)
            def callback(frame, name, func, old_locals):
                rule = parse_rule(
                    rules, func, pred, maker, frame.f_locals, frame.f_globals
                )
                rules.add(rule)
                if old_locals.get(name) in (f, rules):
                    return f    # prevent overwriting if name is the same
                return func
            return decorate_assignment(callback)

        try:
            decorate.__name__ = name
        except TypeError:
            decorate = new.function(
                decorate.func_code, decorate.func_globals, name,
                decorate.func_defaults, decorate.func_closure
            )
        decorate.__doc__ = doc
        return decorate

class RuleSet(object):
    """An observable, stably-ordered collection of rules"""

    #default_action = NoApplicableMethods()
    default_actiontype = Method
    counter = 0

    def __init__(self):
        self.rules = []
        self.actiondefs = {}
        self.listeners = []

    def add(self, rule):
        sequence = self.counter
        self.counter += 1
        actiondefs = frozenset(self._actions_for(rule, sequence))
        self.rules.append( rule )
        self.actiondefs[rule] = sequence, actiondefs
        self.notify(added=actiondefs)

    def remove(self, rule):
        sequence, actiondefs = self.actiondefs.pop(rule)
        self.rules.remove(rule)
        self.notify(removed=actiondefs)

    #def changed(self, rule):
    #    sequence, actions = self.actions[rule]
    #    new_actions = frozenset(self._actions_for(rule, sequence))
    #    self.actions[rule] = sequence, new_actions
    #    self.notify(new_actions-actions, actions-new_actions)

    def notify(self, added=empty, removed=empty):
        for listener in self.listeners:
            listener.actions_changed(added, removed)

    def __iter__(self):
        for rule in self.rules:
            for actiondef in self.actiondefs[rule][1]:
                yield actiondef


    def _actions_for(self, (body, predicate, actiontype), sequence):
        actiontype = actiontype or self.default_actiontype
        for signature in predicate_signatures(predicate):
            yield ActionDef(actiontype, body, signature, sequence)

    def subscribe(self, listener):
        self.listeners.append(listener)
        if self.rules:
            listener.actions_changed(frozenset(self), empty)

    def unsubscribe(self, listener):
        self.listeners.remove(listener)





























class TypeEngine(object):
    """Simple type-based dispatching"""

    def __init__(self, func, rules):
        self.registry = {}
        self.cache = {}
        self.func = func
        self.generate_code()
        rules.subscribe(self)
        self.ruleset = rules

    def actions_changed(self, added, removed):
        registry = self.registry
        for r in removed:
            registry.clear()
            added = self.ruleset
            break   # force full regeneration if any rules removed
        for (atype, body, sig, seq) in added:
            action = atype(body,sig,seq)
            if sig in registry:
                registry[sig] = combine_actions(action, registry[sig])
            else:
                registry[sig] = action
        self.reset_cache()

    def __getitem__(self, types):
        registry = self.registry
        action = self.ruleset.default_action
        for sig in registry:
            if implies(types, sig):
                action = combine_actions(action, registry[sig])
        return action

    def __call__(self, *args):
        # XXX code similar to this should be generated inside self.func
        types = tuple([getattr(arg,'__class__',type(arg)) for arg in args])
        try: f = self.cache[types]
        except KeyError: f = self.cache[types] = self[types]
        return f(*args)


    def generate_code(self):
        c = Code.from_function(self.func, copy_lineno=True)
        args, star, dstar, defaults = inspect.getargspec(self.func)
        types = [
            Call(
                Const(getattr),
                (Local(name), Const('__class__'), Call(Const(type),(Local(name),)))
            ) for name in flatten(args)
        ]
        target = Call(Const(self.cache.get), (tuple(types), Const(self)))
        c.return_(Call(target, map(tuplize,args), (), star, dstar))
        self.func.func_code = c.code()


    def reset_cache(self):

        self.cache.clear()

        if self is implies.__engine__:
            # For every generic function *but* implies(), it's okay to let
            # the cache populate lazily.  But implies() has to be working
            # in order to populate the cache!  Therefore, we have to
            # pre-populate the cache here.  This pre-population must include
            # *only* core framework rules, because rules added later might
            # need to override/combine and so should still be cached lazily.
            #
            if _core_rules:
                # Update from our copy of the core rules
                self.cache.update(_core_rules)
            else:
                # The core is not bootstrapped yet; copy *everything* to cache
                # since anything being added now is a core rule by definition.
                self.cache.update(self.registry)








def flatten(v):
    if isinstance(v,basestring): yield v; return
    for i in v:
        for ii in flatten(i): yield ii

def tuplize(v):
    if isinstance(v,basestring): return Local(v)
    if isinstance(v,list): return tuple(map(tuplize,v))


def rules_for(f, abstract=False):
    if not hasattr(f,'__rules__'):
        f.__rules__ = RuleSet()
        if not abstract:
            clone = new.function(
                f.func_code, f.func_globals, f.func_name, f.func_defaults,
                f.func_closure
            )
            f.__rules__.add(Rule(clone))
    if not hasattr(f, '__engine__'):
        f.__engine__ = TypeEngine(f, f.__rules__)
    return f.__rules__


def abstract():
    """Declare a function to be abstract"""
    def callback(frame, name, func, old_locals):
        rules_for(func, True)
        return func
    return decorate_assignment(callback)


when = Method.make_decorator(
    "when", "Extend a generic function with a new action"
)






abstract()
def implies(s1,s2):
    """Is s2 always true if s1 is true?"""


when(implies, (tuple,tuple))
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
when(implies, (ClassType, type))
def classic_implies_new(s1, s2):
    # A classic class only implies a new-style one if it's ``object``
    # or ``InstanceType``; this is an exception to the general rule that
    # isinstance(X,Y) implies issubclass(X.__class__,Y)
    return s2 is object or s2 is InstanceType

when(implies, (Method,Method))
def method_implies(a1, a2):
    if a1.__class__ is a2.__class__:
        return implies(a1.signature, a2.signature)
    raise TypeError("Incompatible action types", a1, a2)










def YES(s1,s2): return True
def NO(s1,s2):  return False

def always_overrides(a,b):
    """instances of `a` always imply `b`; `b` instances never imply `a`"""
    when(implies, (a, b))(YES)
    when(implies, (b, a))(NO)

def merge_by_default(t):
    """instances of `t` never imply other instances of `t`"""
    when(implies, (t, t))(NO)


def combine_actions(a1,a2):
    """Return a new action for the combination of a1 and a2"""
    if a1 is None:
        return a2
    elif a2 is None:
        return a1
    elif implies(a1,a2):
        if not implies(a2,a1):
            return a1.override(a2)
    elif implies(a2,a1):
        return a2.override(a1)
    return a1.merge(a2)


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

after  = After.make_decorator('after')
merge_by_default(After)

always_overrides(Around, Before)
always_overrides(Around, After)
always_overrides(Before, After)
always_overrides(Before, Method)
always_overrides(After, Method)





class DispatchError(Exception):
    """A dispatch error has occurred"""

    def __call__(self,*args,**kw):
        raise self.__class__(*self.args+(args,kw))  # XXX

merge_by_default(DispatchError)


class NoApplicableMethods(DispatchError):
    """No applicable action has been defined for the given arguments"""

    def merge(self, other):
        return AmbiguousMethods([self,other])

always_overrides(Method, NoApplicableMethods)

RuleSet.default_action = NoApplicableMethods()























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


when(implies, (AmbiguousMethods, Method))
def ambiguous_overrides(a1, a2):
    for m in a1.methods:
        if implies(m, a2):
            # if any ambiguous method overrides a2, we can toss it
            return True
    return False

when(implies, (Method, AmbiguousMethods))
def override_ambiguous(a1, a2):
    for m in a2.methods:
        if not implies(a1, m):
            return False
    return True     # can only override if it overrides all the ambiguity

# needed to disambiguate the above two methods if combining a pair of AM's:
merge_by_default(AmbiguousMethods)

# the core is finished loading, snapshot the core rules for implies()
_core_rules = implies.__engine__.registry.copy()

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



