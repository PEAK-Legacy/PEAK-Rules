from peak.util.assembler import *
from codegen import *
from criteria import *
from predicates import *
from core import *

__all__ = ['Bind', 'match_predicate', 'match_sequence']

nodetype()
def Bind(name, code=None):
    if code is None:
        return name,
    raise TypeError("Can't compile Bind expression")

def match_predicate(pattern, expr, binds):
    """Return predicate matching pattern to expr, updating binds w/bindings"""
    return Test(Comparison(expr), Inequality('==', pattern))

when(match_predicate, (type(None),))
def match_none(pattern, expr, binds):
    return Test(Identity(expr), IsObject(pattern))

when(match_predicate, (Bind,))
def match_bind(pattern, expr, binds):
    if pattern.name != '_':
        vals = binds.setdefault(pattern.name, [])
        if expr not in vals:
            vals.append(expr)
            for old in vals[-2:-1]:
                return Test(Truth(Compare(expr, (('==', old),))), True)
    return True










when(match_predicate, (istype(list),))
when(match_predicate, (istype(tuple),))
def match_sequence(pattern, expr, binds):
   pred = Test(Comparison(Call(Const(len), (expr,))), Value(len(pattern)))
   for pos, item in enumerate(pattern):
       pred = intersect(
           pred, match_predicate(item, Getitem(expr, Const(pos)), binds)
       )
   return pred

when(match_predicate, (Node,))
def match_node(pattern, expr, binds):
   pred = Test(IsInstance(expr), istype(type(pattern)))
   for pos, item in enumerate(pattern):
       if pos:
           pred = intersect(
               pred, match_predicate(item, Getitem(expr, Const(pos)), binds)
           )
   return pred






















