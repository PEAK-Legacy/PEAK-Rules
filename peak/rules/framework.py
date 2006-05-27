__all__ = [
    'Rule', 'RuleSet', 'predicate_signatures', 'abstract', 'when', 'rules_for',
    'implies',
]

from peak.util.decorators import decorate_assignment

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


class Action: pass  # XXX this will be replaced by MethodChain et al






class RuleSet(object):
    """An observable, stably-ordered collection of rules"""

    default_actiontype = Action
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

    def __init__(self, func, rules):
        self.registry = {}
        self.func = func
        # XXX redefine func's code to call us, using BytecodeAssembler
        rules.subscribe(self)
        self.ruleset = rules

    def close(self):
        self.ruleset.unsubscribe(self)

    def actions_changed(self, added, removed):
        for (atype, body, sig, seq) in added:
            self.registry[sig] = body  #XXX atype(body,sig,seq)

    def __getitem__(self, types):
        return self.registry[types]     # XXX

        #registry = self.registry
        #if types in registry:
        #    return registry[types]
        #for sig in registry:
        #    if implies(types, sig):
        #        return registry[sig]    # XXX need precedence, combination
        #raise KeyError(types)

    def __call__(self, *args):
        # XXX should cache - and the code should be generated in the function,
        #     not run here.
        return self[tuple(map(type, args))](*args)










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

from types import ClassType
when(implies, (type,      type)     )(issubclass)
when(implies, (ClassType, ClassType))(issubclass)
when(implies, (type,      ClassType))(issubclass)
when(implies, (ClassType, type)     )(issubclass)   # actually always False!





















