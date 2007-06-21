from __future__ import division
import sys
from peak.rules.core import Aspect, abstract, Dispatching, Engine, when
from peak.util.decorators import struct
from weakref import ref
from sys import maxint
from types import ClassType

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


abstract()
def seeds_for(index, criterion):
    """Determine the seeds needed to index `criterion`

    Methods must return a 3-tuple (seeds, inclusions, exclusions) of seed sets
    or sequences.  See Indexing.txt for details.
    """





class Ordering(Aspect):
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

def define_ordering(ob, seq):
    items = []
    for key in seq:
        Ordering(ob, key).requires(items)
        items.append(key)






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

    def build_root(self, cases, exprs):
        self.memo = {}
        return self.build(cases, exprs)

    def build(self, cases, exprs):
        key = (cases, exprs)
        if key in self.memo:
            return self.memo[key]

        if not exprs:
            node = self.build_leaf(cases)
        else:
            best, rest = self.best_expr(cases, exprs)
            assert len(rest) < len(exprs)

            if best is None:
                # No best expression found, recurse
                node = self.build(cases, rest)
            else:
                node = self.build_node(best, cases, rest)

        self.memo[key] = node
        return node

    def build_node(self, expr, cases, remaining_exprs):
        raise NotImplementedError

    def build_leaf(self, cases):
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

class BitmapIndex(Aspect):
    """Index that computes selectivity and handles basic caching w/bitmaps"""

    known_cases = 0

    def __init__(self, engine, expr):
        self.extra = {}
        self.all_seeds = {}         # seed -> inc_cri, exc_cri
        self.criteria_bits = {}     # cri  -> case bits
        self.case_seeds = []        # case -> seed set
        self.criteria_seeds = {}    # cri  -> seeds, inc_seeds, exc_seeds
        self.expr = expr
        self.engine = engine

    def add_case(self, case_id, criterion):       
        self._extend_cases(case_id)
        if criterion in self.criteria_seeds:
            seeds, inc, exc = self.criteria_seeds[criterion]
        else:
            self.criteria_bits[criterion] = 0
            seeds, inc, exc = self.criteria_seeds[criterion] \
                            = seeds_for(self, criterion)

        self.case_seeds[case_id] = seeds
        bit = to_bits([case_id])
        self.known_cases |= bit
        self.criteria_bits[criterion] |= bit

        all_seeds = self.all_seeds
        for i, seeds in ((0,inc), (1,exc)):
            for seed in seeds:
                if seed not in all_seeds:
                    all_seeds[seed] = set(), set()
                all_seeds[seed][i].add(criterion)

    def _extend_cases(self, case_id):
        if case_id >= len(self.case_seeds):
            self.case_seeds.extend(
                [self.all_seeds]*(case_id+1-len(self.case_seeds))
            )

    def selectivity(self, cases):
        if cases and cases[-1] >= len(self.case_seeds):
            self._extend_cases(cases[-1])
        cases = map(self.case_seeds.__getitem__, cases)
        return (len(self.all_seeds), sum(map(len, cases)))

        '''seeds = -1
        while len(self.all_seeds) > seeds:
            # The loop ensures accuracy in the case where a len() adds seeds
            seeds = len(self.all_seeds)  # must be the value *before* totalling
            total = sum(map(len, cases))            
        return seeds, total'''

    def seed_bits(self, cases):
        bits = self.criteria_bits
        return dict([
            (seed,
                (sum([bits[i] for i in inc]) & cases,
                 sum([bits[e] for e in exc]) & cases))
                for seed, (inc, exc) in self.all_seeds.items()
        ])

    def expanded_sets(self):
        return [
            (seed, [list(from_bits(inc)), list(from_bits(exc))])
            for seed, (inc, exc) in self.seed_bits(self.known_cases).items()
        ]














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


class _FixedSubset(object):
    __slots__ = 'parent'
    def __init__(self, parent):
        self.parent = parent
    def __len__(self):
        return len(self.parent)-1

        
when(seeds_for, (BitmapIndex, Pointer))
def seeds_for_pointer(index, criterion):
    idref = id(criterion.ref)
    if criterion.equal:
        return [idref], [idref], [None]
    return _FixedSubset(index.all_seeds), [None], [idref]






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
def Value(value, truth=True):
    return value, truth

class _RangeSubset(object):
    __slots__ = 'offsets', 'seeds', 'lo', 'hi'

    def __init__(self, index, lo, hi):
        self.offsets = index.extra
        self.seeds   = index.all_seeds
        self.lo = lo
        self.hi = hi

    def __len__(self):
        off = self.offsets
        if not off:
            for n,k in enumerate(sorted(self.seeds)):
                off[k] = n                
        return off[self.hi] - off[self.lo]
        
when(seeds_for, (BitmapIndex, Range))
def seeds_for_range(index, criterion):
    lo, hi = criterion.lo, criterion.hi
    if lo not in index.all_seeds or hi not in index.all_seeds:
        index.extra.clear()   # ensure offsets are rebuilt on next selectivity()
    return _RangeSubset(index, lo, hi), [lo], [hi]

when(seeds_for, (BitmapIndex, Value))
def seeds_for_value(index, criterion):
    v = (criterion.value, 0)
    if v not in index.all_seeds:
        index.extra.clear()   # ensure offsets are rebuilt on next selectivity()
    if criterion.truth:
        return [v], [v], []
    else:
        return _FixedSubset(index.all_seeds), [(Min, -1)], [v]





def split_ranges(ind, cases):
    """Return (exact, ranges) where `exact` is a dict[value]->bits and `ranges`
    is a sorted list of ``((lo,hi),bits)`` tuples expressing non-inclusive
    ranges.  `ind` must be a bitmap index filled with ``Range`` and ``Value``
    criteria, and `cases` must be a bitmap of the cases to be included in the
    result.
    """
    ranges = []
    exact = {}
    current = cases ^ (ind.known_cases & cases)

    bitmap = ind.seed_bits(cases)
    for (val,d), (inc, exc) in bitmap.iteritems():
        if d:
            break     # something other than == was used; use full algorithm
        exact[val] = current | inc
    else:
        return exact, ranges    # all == criteria, no ranges or !=

    low = Min

    for (val,d), (inc, exc) in sorted(bitmap.iteritems()):
        if val != low:
            if ranges and ranges[-1][-1]==current:
                low = ranges.pop()[0][0]
            ranges.append(((low, val), current))
            low = val
        new = current | inc
        new ^= (new & exc)
        if d==0 or d<0 and not isinstance(val, _ExtremeType):
            exact[val] = new
        if d:
            current = new

    if low != Max:
        if ranges and ranges[-1][-1]==current:
            low = ranges.pop()[0][0]
        ranges.append(((low, Max), current))    
    return exact, ranges


