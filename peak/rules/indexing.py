import sys
from peak.rules.core import Aspect, abstract, Dispatching, Engine

try:
    set = set
    frozenset = frozenset
except NameError:
    from sets import Set as set
    from sets import ImmutableSet as frozenset

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
            if i>31:    # under 2.3, this operation does the wrong thing
                i = long(i)
            b |= 1<<i
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

            spread = float(total_cases) / branches
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

