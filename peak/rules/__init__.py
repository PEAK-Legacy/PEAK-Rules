"""The PEAK Rules Framework"""

import peak.rules.core
from peak.rules.core import abstract, when, before, after, around, istype, \
    DispatchError, AmbiguousMethods, NoApplicableMethods, value

def combine_using(*wrappers):
    """Designate a generic function that wraps the iteration of its methods

    Standard "when" methods will be combined by iteration in precedence order,
    and the resulting iterator will be passed to the supplied wrapper(s), last
    first.  (e.g. ``combine_using(sorted, itertools.chain)`` will chain the
    sequences supplied by each method into one giant list, and then sort it).

    As a special case, if you include ``abstract`` in the wrapper list, it
    will be removed, and the decorated function will be marked as abstract.

    This decorator can only be used once per function, and can't be used if
    the generic function already has methods (even the default method) or if
    a custom method type has already been set (e.g. if you already called
    ``combine_using()`` on it before).
    """
    is_abstract = abstract in wrappers
    if is_abstract:
        wrappers = tuple([w for w in wrappers if w is not abstract])
        
    def callback(frame, name, func, old_locals):
        if core.Dispatching.exists_for(func) and list(core.rules_for(func)):
            raise RuntimeError("Methods already defined for", func)
        if is_abstract:
            func = abstract(func)
        r = core.Dispatching(func).rules
        if r.default_actiontype is not core.Method:
            raise RuntimeError("Method type already defined for", func)
        r.default_actiontype = core.MethodList.make
        r.methodlist_wrappers = wrappers[::-1]
        if not is_abstract:
            r.add(core.Rule(core.clone_function(func)))
        return func
    return core.decorate_assignment(callback)

class priority(int):
    """An integer priority for manually resolving a rule ambiguity"""


def let(**kw):
    """Define temporary variables for use in rules and methods

    Usage::

        @when(somefunc, "let(x=foo(y), z=y*2) and x>z")
        def some_method((x,z), next_method, y):
            # do something here

    The keywords used in the let() expression become available for use in
    any part of the rule that is joined to the ``let()`` by an ``and``
    expression, but will not be available in expressions joined by ``or`` or
    ``not`` branches.  Any ``let()`` calls at the top level of the expression
    will also be available for use in the method body, if you place them in
    a tuple argument in the *very first* argument position -- even before
    ``next_method`` and ``self``.

    Note that variables defined by ``let()`` are **lazy** - their values are
    not computed until/unless they are actually needed by the relevant part
    of the rule, so it does not slow things down at runtime to list all your
    variables up front.  Likewise, only the variables actually listed in your
    first-argument tuple are calculated, and only when the method is actually
    invoked.

    (Currently, this feature is mainly to support easy-to-understand rules,
    and DRY method bodies, as variables used in the rule's criteria may be
    calculated a second time when the method is invoked.)

    Note that while variable calculation is lazy, there *is* an evaluation
    order *between* variables in a let; you can't use a let-variable before
    it's been defined; you'll instead get whatever argument, local, or global
    variable would be shadowed by the as-yet-undefined variable.
    """
    raise NotImplementedError("`let` can only be used in rules, not code!")



# TEMPORARY BACKWARDS COMPATIBILITY - PLEASE IMPORT THIS DIRECTLY FROM CORE
# (or better still, use the '>>' operator that method types now have)
#
from peak.rules.core import always_overrides





































