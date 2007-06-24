from peak.rules.core import *
from peak.util.decorators import struct
from weakref import ref
from sys import maxint
__all__ = [
    'Range', 'Value', 'IsObject', 'Min', 'Max', 'Class', 'Classes', 'tests_for',
    'NotObjects',  'Conjunction', 'Disjunction', 'Test', 'Signature', 
]

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

class Intersection(object):
    """Abstract base for conjunctions and signatures"""
    __slots__ = ()

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

struct()
def Value(value, match=True):
    return value, match

when(implies, (Value, Range))(
    # ==Value implies Range if Value is within range
    lambda c1, c2: c1.match and c2.lo <= (c1.value, 0) <= (c2.hi)
)
when(implies, (Range, Value))(
    # Range implies !=Value if Value is out of range
    lambda c1, c2: not c2.match and not (c1.lo <= (c2.value, 0) <= (c1.hi)) or
                   c2.match and c1==Range((c2.value,-1), (c2.value,1))
)
when(implies, (Range, Range))(
    # Range1 implies Range2 if both points are within Range2's bounds
    lambda c1, c2: c1.lo >= c2.lo and c1.hi <= c2.hi
)
when(implies, (Value, Value))(
    lambda c1, c2: c1==c2 or (c1.match and not c2.match and c1.value!=c2.value)
)
when(disjuncts, (Value,))(
    lambda ob: ob.match and [ob] or
               [Range(hi=(ob.value,-1)), Range(lo=(ob.value,1))]
)
when(intersect, (Range, Range))
def intersect_range(c1, c2):
    lo, hi = max(c1.lo, c2.lo), min(c1.hi, c2.hi)
    if hi<=lo:
        return False
    return Range(lo, hi)
when(intersect, (Value, Value))
def intersect_values(c1, c2):
    if not c1.match or not c2.match:
        return intersect(Disjunction([c1]), Disjunction([c2]))
    return False    # no overlap

# if these weren't disjoint, they'd be handled by the implication test of
# intersection; therefore, they must be disjoint (i.e. empty).
when(intersect, (Range, Value))(lambda c1, c2: False)
when(intersect, (Value, Range))(lambda c1, c2: False)

struct()
def Class(cls, match=True):
    return cls, match

when(implies, (Class, Class))
def class_implies(c1, c2):
    if c1==c2:
        # not isinstance(x) implies not isinstance(x) always
        #     isinstance(x) implies isintance(x)      always
        return True
    elif c1.match and c2.match:
        #     isinstance(x) implies     isinstance(y) if issubclass(x,y)
        return implies(c1.cls, c2.cls)
    elif c1.match or c2.match:
        # not isinstance(x) implies     isinstance(x) never
        #     isinstance(x) implies not isinstance(y) never
        return False
    else:
        # not isinstance(x) implies not isinstance(y) if issubclass(y, x)
        return implies(c2.cls, c1.cls)

when(intersect, (Class, Class))(lambda c1,c2: Classes([c1, c2]))


struct()
def Test(expr, criterion):
    return expr, criterion

when(implies, (Test, Test))(
    lambda c1, c2: c1.expr==c2.expr and implies(c1.criterion, c2.criterion)
)
when(disjuncts, (Test,))(
    lambda ob: [Test(ob.expr, d) for d in disjuncts(ob.criterion)]
)
when(intersect, (Test, Test))
def intersect_tests(c1, c2):
    if c1.expr==c2.expr:
        return Test(c1.expr, intersect(c1.criterion, c2.criterion))
    else:
        return Signature([c1, c2])        

class Signature(Intersection, tuple):
    """Represent and-ed Tests, in sequence"""

    def __new__(cls, input):
        output = []
        index = {}
        input = iter(input)
        for new in input:
            if new is True:
                continue
            elif new is False:
                return False
            assert isinstance(new, Test), \
                "Signatures can only contain ``criteria.Test`` instances"
            if new.expr in index:
                posn = index[new.expr]
                old = output[posn]
                if implies(old, new):
                    continue    # 'new' is irrelevant, skip it
                new = output[index[new.expr]] = intersect(old, new)
            else:
                posn = index[new.expr] = len(output)
                output.append(new)

            d = disjuncts(new.criterion)
            if len(d) != 1:
                del output[index[new.expr]]
                return intersect(
                    intersect(Signature(output[:posn]), Disjunction([new])),
                    Signature(output[posn+1:]+list(input))
                )
                
        if not output:
            return True
        elif len(output)==1:
            return output[0]
        return tuple.__new__(cls, output)

    def __repr__(self):
        return "Signature("+repr(list(self))+")"

class IsObject(int):
    """Criterion for 'is' comparisons"""

    __slots__ = 'ref', 'match'

    def __new__(cls, ob, match=True):
        self = IsObject.__base__.__new__(cls, id(ob)&maxint)
        self.match = match
        self.ref = ob
        return self

    def __eq__(self,other):
        return self is other or (
            int(self)==int(other)
            and (not isinstance(other, IsObject) or self.match==other.match)
        )

    def __repr__(self):
        return "IsObject(%r, %r)" % (self.ref, self.match)

when(implies, (IsObject, IsObject))
def implies_objects(c1, c2):
    # c1 implies c2 if it's identical, or if c1=="is x" and c2=="is not y"
    return c1==c2 or (c1.match and not c2.match and c1.ref is not c2.ref)

when(intersect, (IsObject, IsObject))
def intersect_objects(c1, c2):
    #  'is x and is y'            'is not x and is x'
    if (c1.match and c2.match) or (c1.ref is c2.ref):
        return False
    else:
        # "is not x and is not y"
        return NotObjects([c1,c2])








class Disjunction(frozenset):
    """Set of minimal or-ed conditions (i.e. no redundant/implying items)

    Note that a Disjunction can never have less than 2 members, as creating a
    Disjunction with only 1 item returns that item, and creating one with no
    items returns ``False`` (as no acceptable conditions means "never true").
    """
    def __new__(cls, input):
        output = []
        for item in input:
            for new in disjuncts(item):
                for old in output[:]:
                    if implies(new, old):
                        break
                    elif implies(old, new):
                        output.remove(old)
                else:
                    output.append(new)
        if not output:
            return False
        elif len(output)==1:
            return output[0]
        return frozenset.__new__(cls, output)

when(implies, (Disjunction, object))
when(implies, (Disjunction, Disjunction))
def union_implies(c1, c2):  # Or(...) implies x if all its disjuncts imply x
    for c in c1:
        if not implies(c, c2):
            return False
    else:
        return True

when(implies, (object, Disjunction))
def ob_implies_union(c1, c2):   # x implies Or(...) if it implies any disjunct
    for c in c2:
        if implies(c1, c):
            return True
    else:
        return False

# We use @around for these conditions in order to avoid redundant implication
# testing during intersection, as well as to avoid ambiguity with the
# (object, bool) and  (bool, object) rules for intersect().
#
around(intersect, (Disjunction, object))(
    lambda c1, c2: Disjunction([intersect(x,c2) for x in c1])
)
around(intersect, (object, Disjunction))(
    lambda c1, c2: Disjunction([intersect(c1,x) for x in c2])
)
around(intersect, (Disjunction, Disjunction))(
    lambda c1, c2: Disjunction([intersect(x,y) for x in c1 for y in c2])
)

# XXX These rules prevent ambiguity with implies(object, bool) and
# (bool, object), at the expense of redundancy.  This can be cleaned up later
# when we allow cloning of actions for an existing rule.  (That is, when we can
# say "treat (bool, Disjunction) like (bool, object)" without duplicating the
# action.)
#
when(implies, (bool, Disjunction))(lambda c1, c2: not c1)
when(implies, (Disjunction, bool))(lambda c1, c2: c2)

# The disjuncts of a Disjunction are a list of its contents:
when(disjuncts, (Disjunction,))(list)


abstract()
def tests_for(ob):
    """Yield the tests composing ob, if any"""

when(tests_for, (Test,     ))(lambda ob: iter([ob]))
when(tests_for, (Signature,))(lambda ob: iter(ob))
when(tests_for, (bool,     ))(lambda ob: iter([]))







class Conjunction(Intersection, frozenset):
    """Set of minimal and-ed conditions (i.e. no redundant/implied items)

    Note that a Conjunction can never have less than 2 members, as
    creating a Conjunction with only 1 item returns that item, and
    creating one with no items returns ``True`` (since no required conditions
    means "always true").
    """
    def __new__(cls, input):
        output = []
        for new in input:
            for old in output[:]:
                if implies(old, new):
                    break
                elif implies(new, old):
                    output.remove(old)
            else:
                output.append(new)
        if not output:
            return True
        elif len(output)==1:
            return output[0]
        return frozenset.__new__(cls, output)

around(implies, (Intersection, object))
around(implies, (Intersection, Intersection))
def set_implies(c1, c2):
    for c in c1:
        if implies(c, c2):
            return True
    else:
        return False

around(implies, (object, Intersection))
def ob_implies_set(c1, c2):
    for c in c2:
        if not implies(c1, c):
            return False
    else:
        return True

# Intersecting an intersection with something else should return a set of the
# same type as the leftmost intersection.  These methods are declared @around
# to avoid redundant implication testing during intersection, as well as to
# avoid ambiguity with the (object, bool) and  (bool, object) rules for
# intersect().
#
around(intersect, (Intersection, Intersection))(
    lambda c1, c2: type(c1)(list(c1)+list(c2))
)
around(intersect, (Intersection, object))(
    lambda c1, c2: type(c1)(list(c1)+[c2])
)
around(intersect, (object, Intersection))(
    lambda c1, c2: type(c2)([c1]+list(c2))
)

# Default intersection is a Conjunction
when(intersect, (object, object))(lambda c1, c2: Conjunction([c1,c2]))

class Classes(Conjunction):
    """A set of related Class instances"""
    if __debug__:
        def __init__(self, input):
            for item in self:
                assert isinstance(item, Class), \
                    "Classes() items must be ``criteria.Class`` instances"

class NotObjects(Conjunction):
    """Collection of and-ed "is not" conditions"""
    if __debug__:
        def __init__(self, input):
            for item in self:
                assert isinstance(item, IsObject) and not item.match, \
                    "NotObjects() items must be false ``IsObject`` instances"







