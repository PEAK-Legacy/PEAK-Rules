======================
The PEAK Rules Project
======================

This package is very much a work in progress.  I expect it to be very stable
(in the sense of being simple and quirk/bug-free) even in the short term, but
it will also be very *unstable* in the API sense.  That is, the code here
should always be working and fairly robust, but don't be surprised if *your*
code suddenly stops working due to changes here.  Until the API stabilizes,
you won't find a PyPI listing for this code.

.. contents:: **Table of Contents**

---------------------------
Roadmap and Design Overview
---------------------------


Planned API
===========

The ``peak.rules`` package will offer an API that looks something like this:

``@rules.abstract()``
    Decorator to mark a function as abstract, meaning that it has no default
    rule defined.  Various arguments and keyword options will be available to
    provide optimization hints or set the default method combination policy
    for the function.

``@rules.when(f, condition=None)``
    Decorator to extend an existing function (`f`), even if it is not already
    a generic function.  Yes, that's right, you'll be able to extend any
    Python function.  The condition will be optional, which means that if you
    don't specify one, your function will simply replace or wrap the original
    definition.

``@rules.around()``, ``@rules.before()``, ``@rules.after()``
    Just like ``@rules.when()``, except using different combination wrappers.
    As with ``when()``, the condition will be optional and the function doesn't
    have to have been declared "abstract" ahead of time.

Once a function has been made extensible, the usual ``f.when()`` and other
decorators will probably be available, but I'm not 100% decided on that as yet.
Unlike RuleDispatch, PEAK-Rules will have an open-ended method combination
system that doesn't rely on the generic function itself controlling the
combination rules.  So it might be cleaner just to always use ``@around(f, c)``
instead of e.g. ``@f.around(c)``, even though the latter looks a bit more
pleasant to me.

In addition to these functions, there will probably be some exception classes,
and maybe a few other specialty classes or functions, including perhaps some
of the core framework's generic functions.  None of those things are as yet
well-defined enough to specify here.


Development Roadmap
===================

The first versions will focus on developing a core framework for extensible
functions that is itself implemented using extensible functions.  This
self-bootstrapping core will implement a type-tuple-caching engine using
relatively primitive operations, and will then have a method combination
system built on that.  The core will thus be capable of implementing generic
functions with multiple dispatch based on positional argument types, and the
decorator APIs will be built around that.

The next phase of development will add alternative engines that are oriented
towards predicate dispatch and more sophisticated ways of specifying regular
class dispatch (e.g. being able to say things like ``isinstance(x,Foo) or
isinstance(y,Foo)``).  To some extent this will be porting the expression
machinery from RuleDispatch to work on the new core, but in a lot of ways it'll
just be redone from scratch.  Having type-based multiple dispatch available to
implement the framework should enable a significant reduction in the complexity
of the resulting library.

An additional phase will focus on adding new features not possible with the
RuleDispatch engine, such as "predicate functions" (a kind of dynamic macro
or rule expansion feature), "classifiers" (a way of priority-sequencing a
set of alternative criteria) and others.

Finally, specialty features such as index customization, thread-safety,
event-oriented rulesets, and such will be introduced.

There is no defined timeframe for these most of these phases, although I
anticipate that at least the first one will be finished in June.


Design Concepts
===============

Signature
    A condition expressed purely in terms of simple tests "and"ed together,
    using no "or" operations of any kind.

Predicate
    One or more signatures "or"ed together

Rule
    A combination of a predicate, an action type, and a body (usually a
    function.)  The existence of a rule implies the existence of one or more
    actions of the given action type and body, one for each possible signature
    that could match the predicate.

Action Type
    A factory that can produce an Action when supplied with a signature and
    a body.  (Examples in ``peak.rules`` will include the ``MethodList``,
    ``MethodChain``, ``Around``, ``Before``, and ``After`` types.)

Action
    An object representing the behavior of a single invocation of a generic
    function.  Action objects may be combined (using a generic function of
    the form ``combine_actions(a1,a2)``) to create combined methods ala
    RuleDispatch.  Each action comprises at least a signature and a body, but
    actions of more complex types may include other information.

Rule Set
    A collection of rules, combined with some policy information (such
    as the default action type) and optional optimization hints.  A rule
    set does not directly implement dispatching.  Instead, rule engines
    subscribe to rule sets, and the rule set informs them when actions are
    added and removed due to changes in the rule set's rules.

    This would almost be better named an "action set" than a "rule set",
    in that it's (virtually speaking) a collection of actions rather than
    rules.  However, you do add and remove entries from it by specifying
    rules; the actions are merely implied by the rules.

    Generic functions will have a ``__rules__`` attribute that points to their
    rule set, so that the various decorators can add rules to them.  You
    will probably be able to subclass the base RuleSet class or create
    alternate implementations, as might be useful for supporting persistent or
    database-stored rules.  (Although you'd probably also need a custom rule
    engine for that.)

Rule Engine
    An object that manages the dispatching of a given rule set to implement
    a specific generic function.  Generic functions will have an ``__engine__``
    attribute that points to their current engine.  Engines will be responsible
    for doing any indexing, caching, or code generation that may be required to
    implement the resulting generic function.

    The default engine will implement simple type-based multiple dispatch with
    type-tuple caching.  For simple generic functions this is likely to be
    faster than almost anything else, even C-assisted RuleDispatch.  It also
    should have far less definition-time overhead than a RuleDispatch-style
    engine would.

    Engines will be pluggable, and in fact there will be a mechanism to allow
    engines to be switched at runtime when certain conditions are met.  For
    example, the default engine could switch automatically to a
    RuleDispatch-like engine if a rule is added whose conditions can't be
    translated to simple type dispatching.  There will also be some type of
    hint system to allow users to suggest what kind of engine implementation
    or special indexing might be appropriate for a particular function.


------------
Mailing List
------------

Please direct questions regarding this package to the PEAK mailing list; see
http://www.eby-sarna.com/mailman/listinfo/PEAK/ for details.
