try:
    set = set
    frozenset = frozenset
except NameError:
    from sets import Set as set
    from sets import ImmutableSet as frozenset


class AbstractIndex:
    def __init__(self):
        self.constraints = set()

    def add_constraint(self, guards):
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

    def is_legal(self, indexes):
        if not self.constraints:
            return True
        for c in self.constraints:
            for i in c:
                if i in indexes:
                    break
            else:
                return True
        else:
            return False






class TreeBuilder:
    def __init__(self):
        self.memo = {}

    def build(self, cases, indexes):
        key = (cases, indexes)
        if key in self.memo:
            return self.memo[key]

        if not indexes:
            node = self.build_leaf(cases)
        else:
            best, rest = self.best_index(cases, indexes)
            assert len(rest) < len(indexes)

            if best is None:
                # No best index found, recurse
                node = self.build(cases, rest)
            else:
                node = self.build_node(best, cases, rest)

        self.memo[key] = node
        return node

    def build_node(self, index, cases, remaining_indexes):
        raise NotImplementedError

    def build_leaf(self, cases):
        raise NotImplementedError












    def best_index(self, cases, indexes):
        best_index = None
        best_spread = None

        to_do = list(indexes)
        remaining = dict.fromkeys(indexes)
        active_cases = len(cases)
        skipped = []

        while to_do:
            index = to_do.pop()
            if not index.is_legal(remaining):
                # Skip criteria that have unchecked prerequisites
                skipped.append(index)
                continue

            branches, total_cases = index.selectivity(cases)

            if total_cases == active_cases * branches:
                # None of the index keys for this expression eliminate any
                # cases, so this expression isn't needed for dispatching
                del remaining[index]

                # recheck skipped indexes that might be legal now
                to_do.extend(skipped)
                skipped = []
                continue

            spread = float(total_cases) / branches
            if best_index is None or spread < best_spread:
                best_index, best_spread = index, spread
            #    best_cost = self.cost(index, remaining)
            #elif spread==best_spread:
            #    cost = self.cost(index, remaining)
            #    if cost < best_cost:
            #        best_index, best_cost = index, cost

        if best_index is not None:
            del remaining[best_index]
        return best_index, frozenset(remaining)

