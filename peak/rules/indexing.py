from __future__ import division
import sys
from peak.util.addons import AddOn
from peak.rules.core import *
from peak.rules.criteria import *
from peak.rules.core import sorted, set, frozenset
from peak.util.extremes import Min, Max, Extreme
from peak.util.decorators import decorate
from types import InstanceType

__all__ = [
    'Ordering', 'BitmapIndex', 'TreeBuilder', 'split_ranges',
    'to_bits', 'from_bits', 'RangeIndex', 'TypeIndex', 'TruthIndex',
    'PointerIndex', 'bitmap_index_type'
]

def define_ordering(ob, seq):
    items = []
    for key in seq:
        Ordering(ob, key).requires(items)
        items.append(key)




















class Ordering(AddOn):
    """Track inter-expression ordering constraints"""

    def __init__(self, owner, expr):
        self.constraints = set()

    def requires(self, guards):
        c = frozenset(guards)
        cs = self.constraints
        if c in cs:
            return
        for oldc in list(cs):
            if c >= oldc:
                return  # already a less-restrictive condition
            elif oldc >= c:
                cs.remove(oldc)
        cs.add(c)

    def can_precede(self, exprs):
        if not self.constraints:
            return True
        for c in self.constraints:
            for e in c:
                if e in exprs:
                    break
            else:
                return True
        else:
            return False












def to_bits(ints):
    """Return a bitset encoding the numbers contained in sequence `ints`"""
    b = 0
    for i in ints:
        b |= 1<<i
    return b

if sys.version<"2.4":
    def to_bits(ints):
        """Return a bitset encoding the numbers contained in sequence `ints`"""
        b = 0
        for i in ints:
            if i>31:
                i = long(i)
            b |= 1<<i   # under 2.3, this breaks when i>31 unless it's a long
        return b

def from_bits(n):
    """Yield the (ascending) numbers contained in bitset n"""
    b = 0
    while n:
        while not n & 15:
            n >>= 4
            b += 4
        if n & 1:
            yield b
        n >>= 1
        b += 1













class TreeBuilder(object):
    """Template methods for the Chambers&Chen dispatch tree algorithm"""

    def build(self, cases, exprs, memo):
        key = (cases, exprs)
        if key in memo:
            return memo[key]
        if not exprs:
            node = self.build_leaf(cases, memo)
        else:
            best, rest = self.best_expr(cases, exprs)
            assert len(rest) < len(exprs)

            if best is None:
                # No best expression found, recurse
                node = self.build(cases, rest, memo)
            else:
                node = self.build_node(best, cases, rest, memo)
        memo[key] = node
        return node

    def build_node(self, expr, cases, remaining_exprs, memo):
        raise NotImplementedError

    def build_leaf(self, cases, memo):
        raise NotImplementedError

    def selectivity(self, expression, cases):
        """Return (seedcount,totalcases) selectivity statistics"""
        raise NotImplementedError

    def cost(self, expr, remaining_exprs):
        return 1








    def best_expr(self, cases, exprs):
        best_expr = None
        best_spread = None
        to_do = list(exprs)
        remaining = dict.fromkeys(exprs)
        active_cases = len(cases)
        skipped = []

        while to_do:
            expr = to_do.pop()
            if not Ordering(self, expr).can_precede(remaining):
                # Skip criteria that have unchecked prerequisites
                skipped.append(expr)
                continue

            branches, total_cases = self.selectivity(expr, cases)

            if total_cases == active_cases * branches:
                # None of the branches for this expression eliminate any
                # cases, so this expression isn't needed for dispatching
                del remaining[expr]

                # recheck skipped exprs that might be legal now
                to_do.extend(skipped)
                skipped = []
                continue

            spread = total_cases / branches
            if best_expr is None or spread < best_spread:
                best_expr, best_spread = expr, spread
                best_cost = self.cost(expr, remaining)
            elif spread==best_spread:
                cost = self.cost(expr, remaining)
                if cost < best_cost:
                    best_expr, best_cost = expr, cost

        if best_expr is not None:
            del remaining[best_expr]
        return best_expr, frozenset(remaining)


class BitmapIndex(AddOn):
    """Index that computes selectivity and handles basic caching w/bitmaps"""

    known_cases = 0
    match = True,

    decorate(staticmethod)
    def addon_key(*args):
        # Always use BitmapIndex as the add-on key
        return (BitmapIndex,)+args

    def __new__(cls, engine, expr):
        if cls is BitmapIndex:
            cls = bitmap_index_type(engine, expr)
        return super(BitmapIndex, cls).__new__(cls, engine, expr)

    def __init__(self, engine, expr):
        self.extra = {}
        self.null = self.all_seeds = {}    # seed -> inc_cri, exc_cri
        self.criteria_bits = {}             # cri  -> case bits
        self.case_seeds = []                # case -> seed set
        self.criteria_seeds = {}            # cri  -> seeds

    def add_case(self, case_id, criterion):
        if criterion in self.criteria_seeds:
            seeds = self.criteria_seeds[criterion]
        else:
            self.criteria_bits.setdefault(criterion, 0)
            seeds = self.criteria_seeds[criterion] = self.add_criterion(criterion)

        case_seeds = self.case_seeds
        if case_id==len(case_seeds):
            case_seeds.append(seeds)
        else:
            self._extend_cases(case_id)
            case_seeds[case_id] = seeds

        bit = to_bits([case_id])
        self.known_cases |= bit
        self.criteria_bits[criterion] |= bit

    def add_criterion(self, criterion):
        """Ensure `criterion` is indexed, return its "applicable seeds" set
        
        As a side effect, ``self.all_seeds`` must be updated to include any
        new seeds required by `criterion`.
        """
        raise NotImplementedError

    def _extend_cases(self, case_id):
        if case_id >= len(self.case_seeds):
            self.case_seeds.extend(
                [self.null]*(case_id+1-len(self.case_seeds))
            )

    def selectivity(self, cases):
        if cases and cases[-1] >= len(self.case_seeds):
            self._extend_cases(cases[-1])
        cases = map(self.case_seeds.__getitem__, cases)
        return (len(self.all_seeds), sum(map(len, cases)))

    def seed_bits(self, cases):
        bits = self.criteria_bits
        return cases ^ (self.known_cases & cases), dict([
            (seed,
                (sum([bits[i] for i in inc]) & cases,
                 sum([bits[e] for e in exc]) & cases))
                for seed, (inc, exc) in self.all_seeds.items()
        ])

    def expanded_sets(self):
        return [
            (seed, [list(from_bits(inc)), list(from_bits(exc))])
            for seed, (inc, exc) in self.seed_bits(self.known_cases)[1].items()
        ]

    def reseed(self, criterion):
        self.add_criterion(criterion)




    def include(self, seed, criterion, exclude=False):
        try:
            s = self.all_seeds[seed]
        except KeyError:
            s = self.all_seeds[seed] = set(), set()
        s[exclude].add(criterion)
        self.criteria_bits.setdefault(criterion, 0)

    def exclude(self, seed, criterion):
        self.include(seed, criterion, True)



class _ExclusionSet(object):
    __slots__ = 'parent', 'exclude'

    def __init__(self, parent, exclude):
        self.parent = parent
        self.exclude = exclude

    def __iter__(self):
        exclude = self.exclude
        for item in self.parent:
            if item!=exclude:
                yield item

    def __len__(self):
        return len(self.parent)-1













def split_ranges(dont_cares, bitmap, node=lambda b:b):
    """Return (exact, ranges) where `exact` is a dict[value]->node and `ranges`
    is a sorted list of ``((lo,hi),node)`` tuples expressing non-inclusive
    ranges.  `dont_cares` and `seedmap` should be the return values from
    a bitmap index's ``seed_bits()`` method.  `node(bits)` should return
    the value to be used as a node in the output; the default is to just return
    a bitmap of cases.
    """
    ranges = []
    exact = {}
    current = dont_cares
    for (val,d), (inc, exc) in bitmap.iteritems():
        if d and not (val is Min and d==-1 and not inc):
            break     # something other than == was used; use full algorithm
        exact[val] = node(current | inc)
    else:
        return exact, [((Min, Max), node(dont_cares))]

    low = Min

    for (val,d), (inc, exc) in sorted(bitmap.iteritems()):
        if val != low:
            if ranges and ranges[-1][-1]==current:
                low = ranges.pop()[0][0]
            ranges.append(((low, val), node(current)))
            low = val
        new = current | inc
        new ^= (new & exc)
        if not isinstance(val, Extreme):
            if d==0 or d<0:
                exact[val] = node(new)
            elif val not in exact:
                exact[val] = node(current)
        if d:
            current = new
    if low != Max:
        if ranges and ranges[-1][-1]==current:
            low = ranges.pop()[0][0]
        ranges.append(((low, Max), node(current)))
    return exact, ranges

class PointerIndex(BitmapIndex):
    """Index for pointer equality"""

    def add_criterion(self, criterion):
        seed = id(criterion.ref)
        self.extra[seed] = 1
        if criterion.match:
            self.include(seed, criterion)
            self.exclude(None, criterion)
            return self.match
        else:
            self.include(None, criterion)
            self.exclude(seed, criterion)
            return self.extra


class TruthIndex(BitmapIndex):
    """Index for simple boolean expression tests"""

    def add_criterion(self, criterion):
        assert isinstance(criterion, Value) and criterion.value is True
        self.include(criterion.match, criterion)
        self.include(not criterion.match, not criterion)
        return self.match


def _get_mro(cls):
    if isinstance(cls, type):
        mro = cls.__mro__
    else:
        class tmp(cls, object): pass
        mro = tmp.__mro__[1:-1] + (InstanceType, object)
    return mro

abstract()
def bitmap_index_type(engine, expr):
    """Get the BitmapIndex subclass to use for the given engine and `expr`"""
    raise NotImplementedError(engine, expr)



class TypeIndex(BitmapIndex):
    """Index for istype(), Class(), and Classes() criteria"""

    def add_class(self, cls):
        if cls in self.all_seeds:
            return

        t = istype(cls)
        self.include(cls, t)    # exact type match

        csmap = self.criteria_seeds
        intersects = self.extra
        all_seeds = self.all_seeds

        for base in _get_mro(cls):
            if base not in all_seeds:
                self.add_class(base)

            self.exclude(cls, Class(base, False))   # issubclass non-match

            c = Class(base)
            if c not in csmap:
                csmap[c] = set()
            csmap[c].add(cls)     # issubclass match
            self.include(cls, c)  # issubclass match

            if base in intersects:
                # multi-class matches
                for c2 in intersects[base]:
                    if implies(t, c2):
                        self.include(cls, c2)
                        csmap[c2].add(cls)

    def seed_bits(self, cases):
        bits = self.criteria_bits
        io = istype(object)
        dontcares, seedbits = super(TypeIndex, self).seed_bits(cases)
        for cri in self.all_seeds[object][0]:
            if cri!=io: dontcares |= bits[cri]
        return dontcares & cases, seedbits

    def add_criterion(self, criterion):
        if isinstance(criterion, istype):
            self.add_class(criterion.type)
            if criterion.match:
                return self.match
            else:
                self.include(object, criterion)
                self.exclude(criterion.type, criterion)
                return _ExclusionSet(self.all_seeds, criterion.type)

        elif isinstance(criterion, Class):
            self.add_class(criterion.cls)
            if criterion.match:
                return self.criteria_seeds[criterion]
            else:
                self.include(object, criterion)
                return _DiffSet(
                    self.all_seeds, self.criteria_seeds[Class(criterion.cls)]
                )

        elif isinstance(criterion, Classes):
            t, m = object, 1    # find Class() w/longest mro to use as target
            for c in criterion:
                self.add_criterion(c)
                if isinstance(c, Class) and c.match:
                    n = len(_get_mro(c.cls))
                    if n > m:
                        t, m = c.cls, n
            seeds = set()
            for cls in self.all_seeds:
                # only look within the relevant subtree
                if t is object or issubclass(cls, t):
                    if implies(istype(cls), criterion):
                        seeds.add(cls)
                        self.include(cls, criterion)
            self.extra.setdefault(t, set()).add(criterion)
            return seeds

        raise TypeError(criterion)


class RangeIndex(BitmapIndex):

    def __init__(self, engine, expr):
        BitmapIndex.__init__(self, engine, expr)
        self.null = None

    def add_criterion(self, criterion):

        if isinstance(criterion, Range):
            lo = i = criterion.lo
            hi = e = criterion.hi
            match = True

        elif isinstance(criterion, Value):
            lo = hi = val = (criterion.value, 0)
            if criterion.match:
                e = (Min, -1)
                i = val
            else:
                i = (Min, -1)
                e = val
            match = criterion.match

        else:
            raise TypeError(criterion)

        if (i not in self.all_seeds or e not in self.all_seeds
            or (lo,hi,match) not in self.extra
        ):
            # oops, there's a seed we haven't seen before:
            # make sure the offsets are rebuilt on the next selectivity() call
            self.extra.clear()   

        self.include(i, criterion)
        self.exclude(e, criterion)
        return lo, hi, match





    def selectivity(self, cases):
        case_seeds = self.case_seeds
        if cases and cases[-1] >= len(case_seeds):
            self._extend_cases(cases[-1])

        extras = self.extra
        if not extras:
            all_seeds = self.all_seeds
            offsets = dict([(k,n) for n,k in enumerate(sorted(all_seeds))])
            all = extras[None] = len(all_seeds)
            all_but_one = all - 1  
            for lo,hi,inc in self.criteria_seeds.values():
                if lo==hi:
                    extras[lo,hi,inc] = [all_but_one, 1][inc]
                else:
                    extras[lo,hi,inc] = offsets[hi] - offsets[lo]

        cases = map(case_seeds.__getitem__, cases)
        return (len(self.all_seeds), sum(map(extras.__getitem__, cases)))


class _DiffSet(object):
    def __init__(self, base, subtract):
        self.base = base
        self.subtract = subtract
        self.baselen = -1
        self.cache = None

    def __len__(self):
        if len(self.base) != self.baselen:
            self.cache = set(self.base) - self.subtract
            self.baselen = len(self.base)
        return len(self.cache)

    def __iter__(self):
        if len(self.base) != self.baselen:
            len(self)   # recache
        return iter(self.cache)



