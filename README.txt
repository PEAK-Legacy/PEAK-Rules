======================
The PEAK Rules Project
======================

This package is very much a work in progress.  I expect it to be very stable
(in the sense of being simple and quirk/bug-free) even in the short term, but
it will also be very *unstable* in the API sense.  That is, the code here
should always be working and fairly robust, but don't be surprised if *your*
code suddenly stops working due to API changes!


----------
QuickStart
----------

Installation::

    easy_install svn://svn.eby-sarna.com/svnroot/BytecodeAssembler
    easy_install svn://svn.eby-sarna.com/svnroot/PEAK-Rules

Usage::

    from peak.rules import abstract, when, around, before, after

    @abstract()
    def pprint(ob):
        """A pretty-printing generic function"""

    @when(pprint, (list,))
    def pprint_list(ob):
        # etc...

Basically, at the present moment, PEAK-Rules supports multiple-dispatch on
positional arguments by *type only*.  But it supports the full method
combination semantics of RuleDispatch using a new decentralized approach,
that allows you to easily create new method types or combination semantics,
complete with their own decorators (like ``when``, ``around``, etc.)

These decorators also all work with *existing* functions; you do not have to
predeclare a function generic in order to use it.  You can also omit the
condition from the decorator call, in which case the effect is the same as
RuleDispatch's ``strategy.default``, i.e. there is no condition.  Thus, you
can actually use PEAK-Rules's ``around()`` as a quick way to monkeypatch
existing functions, even ones defined by other packages.  And the decorators
use the ``DecoratorTools`` package, so you can omit the ``@`` signs for
Python 2.3 compatibility.

Currently, the only conditions you can give to the decorators are tuples of
types -- or objects that you've created and defined an ``implies()``
relationship between them and a tuple of types!

``peak.rules.implies()`` is the generic function that's used to define
implication relationships, and it is user-extensible.  The current rule engine
only works with type tuples, though, so you're limited in what you can do with
it.


-----------------------
Where All This Is Going
-----------------------

The big differences between PEAK-Rules and RuleDispatch are:

1. It's designed for extensibility/pluggability from the ground up

2. It's built without adaptation, only generic functions, and so doesn't carry
   as much baggage.  (The current implementation is about 1500 lines of code:
   the size of just one of RuleDispatch's modules.)

While it's true that the current default rule engine doesn't support arbitrary
predicates, the point is that it's *pluggable*.  Future versions of PEAK-Rules
will include another engine similar to the one in RuleDispatch, and when that
happens the current engine will automatically switch over when it encounters
rules it can't handle.  This means that PEAK-Rules can use custom-tuned engines
for specific application scenarios, and over time it will evolve the ability
to accept "tuning hints" to adjust the indexing techniques for special cases.

What got me started on all this was Guido's super-small multimethod prototype
for Python 3000.  It was simple enough and fast enough that it got me thinking
it was good enough for maybe 50% of what you need generic functions for,
especially if you added method combination.  RuleDispatch was always conceived
as a single implementation of a single dispatch algorithm intended to be
"good enough" for all uses.

Guido's argument on the Py3K mailing list, however, was that applications with
custom dispatch needs should write custom dispatchers.  And I almost agree --
except that I think they should get a RuleDispatch-like dispatcher for free,
and be able to tune or write ones to plug in for specialized needs.  And thus,
the idea of PEAK-Rules.

The kicker was that Guido's experiment with type-tuple caching (a predecessor
algorithm to the Chambers-and-Chen algorithm used by RuleDispatch) showed it to
be fast *enough* for common uses, even without any C code, as long as you were
willing to do a little code generation.  And so, here it is.  Type-tuple-cached
multiple dispatch with method combination.  It's not quite CLOS and it's sure
not RuleDispatch, but it's a solid foundation for porting the rest of
RuleDispatch's functionality.

There's a heck of a lot of work left to do to implement that port.  The good
news, though, is that a lot of the algorithms will be simpler, thanks to
the new core.  Many times while writing RuleDispatch I wished I had a generic
function engine already in existence, so I could use generic functions to write
it with.  Now I have that, so it should be easier.

Still, there's about 2500 lines of code (not counting tests) that will need
reworking.  I don't plan on dumping anything over except dispatch.ast_builder
and its tests; it's purely a Python expression compiler library and not in any
way specific to RuleDispatch.  The other stuff, dispatch.functions,
dispatch.predicates, and dispatch.strategy are going to probably get redone in
some fairly fundamental ways.


------------
Mailing List
------------

Please direct questions regarding this package to the PEAK mailing list; see
http://www.eby-sarna.com/mailman/listinfo/PEAK/ for details.
