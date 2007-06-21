from peak.rules.core import Aspect, abstract, when
from peak.util.decorators import struct
from weakref import ref
from sys import maxint

__all__ = [
    'Range', 'Value', 'Pointer', 'Min', 'Max', 'Class', 'Classes',
    # 'intersect', 'Signature', 'Predicate', ...
]

try:
    set = set
    frozenset = frozenset
except NameError:
    from sets import Set as set
    from sets import ImmutableSet as frozenset


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











class _ExtremeType(object):     # Courtesy of PEP 326
    def __init__(self, cmpr, rep):
        object.__init__(self)
        self._cmpr = cmpr
        self._rep = rep

    def __cmp__(self, other):
        if isinstance(other, self.__class__) and\
           other._cmpr == self._cmpr:
            return 0
        return self._cmpr

    def __repr__(self):
        return self._rep

    def __lt__(self,other):
        return self.__cmp__(other)<0

    def __le__(self,other):
        return self.__cmp__(other)<=0

    def __gt__(self,other):
        return self.__cmp__(other)>0

    def __eq__(self,other):
        return self.__cmp__(other)==0

    def __ge__(self,other):
        return self.__cmp__(other)>=0

    def __ne__(self,other):
        return self.__cmp__(other)<>0

Max = _ExtremeType(1, "Max")
Min = _ExtremeType(-1, "Min")

struct()
def Range(lo=(Min,-1), hi=(Max,1)):
    assert hi>lo
    return lo, hi

class Pointer(int):
    """Criterion for 'is' comparisons"""

    __slots__ = 'ref', 'equal'

    def __new__(cls, ob, equal=True):
        self = Pointer.__base__.__new__(cls, id(ob)&maxint)
        self.equal = equal
        self.ref = ob
        return self

    def __eq__(self,other):
        return self is other or (
            int(self)==int(other)
            and (not isinstance(other, Pointer) or self.equal==other.equal)
        )

    def __repr__(self):
        return "Pointer(%r)" % self.ref


struct()
def Value(value, truth=True):
    return value, truth


struct()
def Class(cls, match=True):
    return cls, match


class Classes(frozenset):
    """A set of related Class instances"""
    # XXX reduce inclusions to most-specific, exclusions to least-specific
    






