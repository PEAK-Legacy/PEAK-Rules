from __future__ import division
import sys
from peak.util.addons import AddOn
from peak.rules.core import abstract, Dispatching, Engine, when
from peak.rules.criteria import *
from peak.rules.criteria import sorted, set, frozenset
from peak.util.extremes import Min, Max, Extreme

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




def always_testable(expr):
    """Is `expr` safe to evaluate in any order?"""
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

class BitmapIndex(AddOn):
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
        if seeds is not self.all_seeds: self.known_cases |= bit
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


abstract()
def seeds_for(index, criterion):
    """Determine the seeds needed to index `criterion`

    Methods must return a 3-tuple (seeds, inclusions, exclusions) of seed sets
    or sequences.  See Indexing.txt for details.
    """

when(seeds_for, (BitmapIndex, bool))(
    # True->all seeds, False->no seeds
    lambda index, criterion: ([(), index.all_seeds][criterion], [], [])
)







class _FixedSubset(object):
    __slots__ = 'parent'
    def __init__(self, parent):
        self.parent = parent
    def __len__(self):
        return len(self.parent)-1

        
when(seeds_for, (BitmapIndex, IsObject))
def seeds_for_pointer(index, criterion):
    idref = id(criterion.ref)
    if criterion.match:
        return [idref], [idref], [None]
    return _FixedSubset(index.all_seeds), [None], [idref]


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
    if criterion.match:
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
        if d==0 or d<0 and not isinstance(val, Extreme):
            exact[val] = new
        if d:
            current = new

    if low != Max:
        if ranges and ranges[-1][-1]==current:
            low = ranges.pop()[0][0]
        ranges.append(((low, Max), current))    
    return exact, ranges


when(seeds_for, (BitmapIndex, Class))
def seeds_for_class(index, criterion):

    cls = criterion.cls
    if isinstance(cls, type):
        mro = cls.__mro__
    else:
        class tmp(cls, object): pass
        mro = tmp.__mro__[1:]

    parents = []
    csmap = index.criteria_seeds
    all_seeds = index.all_seeds
    unions = index.extra
    
    for base in mro:
        parents.append(base)
        c = Class(base)
        if c not in csmap:
            csmap[c] = set(
                # ancestors aren't always parents of things past them in mro
                [p for p in parents if issubclass(p, base)]
            ), set([base]), []
        else:
            csmap[c][0].add(cls)

        if base not in all_seeds:
            all_seeds[base] = set(), set()

        if base in unions:
            for s in unions[base]: s.add(cls)

    if not criterion.match:
        return _DiffSet(
            index.all_seeds, csmap[Class(cls)][0]
        ), [object], [cls]

    return csmap[criterion]



when(seeds_for, (BitmapIndex, Classes))
def seeds_for_classes(index, criterion):

    csmap = index.criteria_seeds
    excluded, required = sets = [], []

    for c in criterion:
        if c not in csmap:
            csmap.setdefault(c, seeds_for(index, c))
        sets[c.match].append(c)

    ex_classes = [c.cls for c in excluded]
    cex = Classes(excluded)
    if cex not in csmap:
        ex_union = reduce(
            set.union, [csmap[c][0].subtract for c in excluded], set()
        )
        for c in ex_classes:
            index.extra.setdefault(c, []).append(ex_union)
        
        csmap[cex] = _DiffSet(index.all_seeds, ex_union), [object], ex_classes

    if required:
        required = [csmap[c][0] for c in required] or index.all
        required = _MultiSet(index, criterion, required, csmap[cex][0].subtract)
        return required, [], ex_classes

    return csmap[cex]













class _MultiSet(object):
    def __init__(self, index, classes, required, excluded):
        self.all_seeds = index.all_seeds
        self.classes = classes
        self.lastlen = self.cachelen = 0
        self.seen = set()
        self.required = required
        self.excluded = excluded

    def __len__(self):
        if len(self.all_seeds)==self.lastlen:
            return self.cachelen        
        s = reduce(set.intersection, self.required) - self.excluded
        l = self.cachelen = len(s)
        self.lastlen = len(self.all_seeds)
        if l > len(self.seen):
            for cls in s - self.seen:
                for c in cls.__bases__:
                    if c in s:
                        break   # not a root if any of its bases are in the set
                else:
                    # Flag the new root as an inclusion point for our criterion
                    self.all_seeds[cls][0].add(self.classes)
            self.seen = s
        return l

class _DiffSet(object):   
    def __init__(self, base, subtract):
        self.base = base
        self.subtract = subtract
        self.baselen = -1
        self.cache = None

    def __len__(self):        
        if len(self.base)>self.baselen:
            self.cache = set(self.base) - self.subtract
            self.baselen = len(self.base)
        return len(self.cache)



