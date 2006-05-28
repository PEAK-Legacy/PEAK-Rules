__all__ = [
    'Rule', 'RuleSet', 'predicate_signatures', 'abstract', 'when', 'rules_for',
    'implies', 'combine_actions', 'Method', 'Around', 'AmbiguousMethods',
    'NoApplicableMethods',
]

from peak.util.decorators import decorate_assignment, decorate
import inspect

try:
    set, frozenset
except NameError:
    from sets import Set as set
    from sets import ImmutableSet as frozenset

empty = frozenset()


class Rule(object):
    """A Rule"""

    __slots__ = "body", "predicate", "actiontype"

    def __init__(self, body, predicate=(), actiontype=None):
        self.body = body
        self.predicate = predicate
        self.actiontype = actiontype

    def __repr__(self):
        return "Rule%r" % ((self.body, self.predicate, self.actiontype),)


def predicate_signatures(predicate):
    yield predicate # XXX







class Method:
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

    #def merge(self, other): return Ambig


class RuleSet(object):
    """An observable, stably-ordered collection of rules"""

    default_actiontype = Method
    counter = 0

    def __init__(self):
        self.rules = []
        self.actiondefs = {}
        self.listeners = []

    def add(self, rule):
        sequence = self.counter
        self.counter += 1
        actiondefs = frozenset(self.actions_for(rule, sequence))
        self.rules.append( rule )
        self.actiondefs[rule] = sequence, actiondefs
        self.notify(added=actiondefs)

    def remove(self, rule):
        sequence, actiondefs = self.actiondefs.pop(rule)
        self.rules.remove(rule)
        self.notify(removed=actiondefs)

    #def changed(self, rule):
    #    sequence, actions = self.actions[rule]
    #    new_actions = frozenset(self.actions_for(rule, sequence))
    #    self.actions[rule] = sequence, new_actions
    #    self.notify(new_actions-actions, actions-new_actions)

    def notify(self, added=empty, removed=empty):
        for listener in self.listeners:
            listener.actions_changed(added, removed)

    def __iter__(self):
        for rule in self.rules:
            for actiondef in self.actiondefs[rule][1]:
                yield actiondef



    def actions_for(self, rule, sequence):
        actiontype = rule.actiontype or self.default_actiontype
        for signature in predicate_signatures(rule.predicate):
            yield (actiontype, rule.body, signature, sequence)

    def subscribe(self, listener):
        self.listeners.append(listener)
        listener.actions_changed(frozenset(self), empty)

    def unsubscribe(self, listener):
        self.listeners.remove(listener)






























class DefaultEngine(object):
    """Simple type-based dispatching"""

    def __init__(self, func, rules):
        self.registry = {}
        self.func = func
        # XXX redefine func's code to call us, using BytecodeAssembler
        rules.subscribe(self)
        self.ruleset = rules

    def close(self):
        self.ruleset.unsubscribe(self)

    def actions_changed(self, added, removed):
        # XXX support removes
        for (atype, body, sig, seq) in added:
            # XXX needs to be a list, to allow combinations for a signature
            self.registry[sig] = atype(body,sig,seq)

    def __getitem__(self, types):
        registry = self.registry

        if types in registry:   # XXX shouldn't be needed w/a populated cache
            return registry[types]

        action = NoApplicableMethods()
        for sig in registry:
            if implies(types, sig):
                # XXX this should pull from the cache instead
                action = combine_actions(action, registry[sig])
        return action

    def __call__(self, *args):
        # XXX should cache - and the code should be generated in the function,
        #     not run here.
        return self[
            tuple([getattr(arg,'__class__',type(arg)) for arg in args])
        ](*args)



def rules_for(f, abstract=True):
    if not hasattr(f,'__rules__'):
        f.__rules__ = RuleSet()
    if not hasattr(f, '__engine__'):
        f.__engine__ = DefaultEngine(f, f.__rules__)
    return f.__rules__


def abstract():
    """Declare a function to be abstract"""
    def callback(frame, name, func, old_locals):
        rules_for(func)
        return func
    return decorate_assignment(callback)


def when(f,pred=()):
    """Extend a generic function with a new action"""

    rules = rules_for(f)

    def callback(frame, name, func, old_locals):
        rules.add( Rule(func, pred) )
        if old_locals.get(name) in (f, rules):
            return f    # prevent overwriting if name is the same
        return func

    return decorate_assignment(callback)













abstract()
def implies(s1,s2):
    """Is s2 always true if s1 is true?"""
    return implies.__engine__(s1,s2)    # XXX until we get code generation

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





























'''class MethodList(Method):
    """A list of related methods"""

    def __init__(self, items=(), tail=None):
        self.items = list(items)
        self.tail = tail
        self.can_tail = True

    decorate(classmethod)
    def make(cls, body, signature=(), precedence=0):
        return cls( [(signature, precedence, body)] )

    def __repr__(self):
        data = self.bodies, self.tail
        return self.__class__.__name__+repr(data)

    def tail_with(self, tail):
        return self.__class__(self.bodies, tail)

    def merge(self, other):
        if other.__class__ is not self.__class__:
            raise TypeError("Incompatible action types for merge", self, other)
        return self.__class__(
            self.bodies+other.bodies, combine_actions(self.tail, other.tail)
        )
'''















class DispatchError(Exception):
    """A dispatch error has occurred"""

    def __call__(self,*args,**kw):
        raise self.__class__(*self.args+(args,kw))  # XXX


class AmbiguousMethods(DispatchError):
    """More than one choice of action is possible"""

    def __init__(self, actions):
        self.actions = actions


class NoApplicableMethods(DispatchError):
    """No applicable action has been defined for the given arguments"""

YES = lambda s1,s2: True
NO  = lambda s1,s2: False

when(implies, (Method, NoApplicableMethods))(YES)
when(implies, (NoApplicableMethods, Method))(NO)
when(implies, (NoApplicableMethods, NoApplicableMethods))(NO)

class Around(Method):
    """'Around' Method (takes precedence over regular methods)"""

when(implies, (Around,Method))(YES)
when(implies, (Method,Around))(NO)












