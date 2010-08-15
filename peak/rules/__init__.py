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

