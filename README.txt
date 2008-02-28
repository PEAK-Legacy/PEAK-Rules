===========================================
Creating Generic Functions using PEAK-Rules
===========================================

PEAK-Rules is a highly-extensible framework for creating and using generic
functions, from the very simple to the very complex.  Out of the box, it
supports multiple-dispatch on positional arguments using tuples of types,
full predicate dispatch using strings containing Python expressions, and
CLOS-like method combining.  (But the framework allows you to mix and match
dispatch engines and custom method combinations, if you need or want to.)

Basic usage::

    >>> from peak.rules import abstract, when, around, before, after

    >>> @abstract()
    ... def pprint(ob):
    ...     """A pretty-printing generic function"""

    >>> @when(pprint, (list,))
    ... def pprint_list(ob):
    ...     print "pretty-printing a list"

    >>> @when(pprint, "isinstance(ob,list) and len(ob)>50")
    ... def pprint_long_list(ob):
    ...     print "pretty-printing a long list"

    >>> pprint([1,2,3])
    pretty-printing a list

    >>> pprint([42]*1000)
    pretty-printing a long list

    >>> pprint(42)
    Traceback (most recent call last):
      ...
    NoApplicableMethods: ...

PEAK-Rules works with Python 2.3 and up -- just omit the ``@`` signs if your
code needs to run under 2.3.  Also, note that with PEAK-Rules, *any* function
can be generic: you don't have to predeclare a function as generic.  (The
``abstract`` decorator is used to declare a function with no *default* method.)

PEAK-Rules is still under development; it lacks much in the way of error
checking, so if you mess up your rules, it may not be obvious where or how you
did.  User documentation is also lacking, although there are extensive doctests
describing most of its internals.

Source distribution snapshots are generated daily, but you can also update
directly from the `development version`_ in SVN.

.. _development version: svn://svn.eby-sarna.com/svnroot/PEAK-Rules#egg=PEAK_Rules-dev
.. _toc:

.. contents:: **Table of Contents**

-----------------
Developer's Guide
-----------------

XXX basics tutorial should go here


Method Combination and Custom Method Types
==========================================

Sometimes, more than one method of a generic function applies in a given
circumstance.  For example, you might need to sum the results of a series of
pricing rules in order to compute a product's price.  Or, sometimes you'd like
a method to be able to modify the result of a less-specific method.

For these scenarios, you will want to use "method combination", either using
PEAK-Rules' built-in method decorators, or custom method types of your own.


Using ``next_method``
---------------------

By default, a generic function will only invoke the most-specific applicable
method.  However, if you add a ``next_method`` argument to the beginning of
an individual method's signature, you can use it to call the "next method"
that applies.  That is, the second-most-specific method.  If that method also
has a ``next_method`` argument, it too will be able to invoke the next method
after it, and so on, down through all the applicable methods.  For example::

    >>> from peak.rules import DispatchError

    >>> @abstract()
    ... def foo(bar, baz):
    ...     """Foo bar and baz"""

    >>> @when(foo, "bar>1 and baz=='spam'")
    ... def foo_one_spam(next_method, bar, baz):
    ...     return bar + next_method(bar, baz)

    >>> @when(foo, "baz=='spam'")
    ... def foo_spam(bar, baz):
    ...     return 42

    >>> @when(foo, "baz=='blue'")
    ... def foo_spam(next_method, bar, baz):
    ...     # if next_method is an instance of DispatchError, it means
    ...     # that calling it will raise that error (NoApplicableMethods
    ...     # or AmbiguousMethods)
    ...     assert isinstance(next_method, DispatchError)
    ... 
    ...     # but we'll call it anyway, just to demo the error
    ...     return 22 + next_method(bar, baz)

    >>> foo(2,"spam")   # 2 + 42
    44

    >>> foo(2,"blue")   # 22 + ...no next method!
    Traceback (most recent call last):
      File ... combiners.txt... in foo_spam
        return 22 + next_method(self,bar,baz)
    ...
    NoApplicableMethods: ...

Notice that ``next_method`` comes *before* ``self`` in the arguments if the
generic function is an instance method.  (If used, it must be the *very first*
argument of the method.)  Its value is supplied automatically by the generic
function machinery, so when you call ``next_method`` you do not have to care
whether the next method needs to know *its* next method; just pass in all of
the *other* arguments (including ``self`` if applicable) and the
``next_method`` implementation will do the rest.

Also notice that methods that do not call their next method do not need to have
a ``next_method`` argument.  If a method calls ``next_method`` when there are
no further methods available, ``NoApplicableMethods`` is raised.  Similarly,
if there is more than one "next method" and they are all equally specific
(i.e. ambiguous), then ``AmbiguousMethods`` is raised.

Most of the time, you will know when writing a routine whether it's safe to
call ``next_method``.  But sometimes you need a routine to behave differently
depending on whether a next method is available.  If calling ``next_method``
will raise an error, then ``next_method`` will be an instance of the error
class, so you can detect it with ``isinstance()``.  If there are no remaining
methods, then ``next_method`` will be an instance of ``NoApplicableMethods``,
and if the next method is ambiguous, it will be an ``AmbiguousMethods``
instance.  In either case, calling ``next_method`` will raise that error with
the supplied arguments.  (And ``DispatchError`` is a base class of both
``AmbiguousMethods`` and ``NoApplicableMethods``, so you can just check for
that.)


Before/After Methods
--------------------

Sometimes you'd like for some additional validation or notification to occur
before or after the "normal" or "primary" methods.  This is what "before",
"after", and "around" methods are for.  For example::

    >>> class BankAccount:
    ...
    ...     def __init__(self,balance,protection=0):
    ...         self.balance = balance
    ...         self.protection = protection
    ...
    ...     def withdraw(self,amount):
    ...         """Withdraw 'amount' from bank"""
    ...         self.balance -= amount      # nominal case
    ...
    ...     @before(withdraw, "amount>self.balance and self.protection==0")
    ...     def prevent_overdraft(self, amount):
    ...         raise ValueError("Insufficient funds")
    ...
    ...     @after(withdraw, "amount>self.balance")
    ...     def automatic_overdraft(self, amount):
    ...         print "Transferring",-self.balance,"from overdraft protection"
    ...         self.protection += self.balance
    ...         self.balance = 0

    >>> acct = BankAccount(200)
    >>> acct.withdraw(400)
    Traceback (most recent call last):
    ...
    ValueError: Insufficient funds

    >>> acct.protection = 300
    >>> acct.withdraw(400)
    Transferring 200 from overdraft protection
    >>> acct.balance
    0
    >>> acct.protection
    100

This specific example could have been written entirely with normal ``when()``
methods, by using more complex conditions.  But, in more complex scenarios,
where different modules may be adding rules to the same generic function, it's
not possible for one module to predict whether its conditions will be more
specific than another's, and whether it will need to call ``next_method``, etc.

So, generic functions offer ``before()`` and ``after()`` methods, that run
before and after the ``when()`` (aka "primary") methods, respectively.  Unlike
primary methods, ``before()`` and ``after()`` methods:

* Are allowed to have ambiguous conditions (and if they do, they execute in the
  order in which they were added to the generic function)

* Are *always* run when their conditions apply, with no need to call
  ``next_method`` to invoke the next method

* Cannot return a useful value and do not have access to the return value of
  any other method

The overall order of method execution is:

1. All applicable ``before()`` methods, from most-specific to least-specific,
   methods at the same level of specificity execute in the order they were
   added.

2. Most-specifc primary method, which may optionally chain to less-specific
   primary methods.  ``AmbiguousMethods`` or ``NoApplicableMethods`` may be
   raised if the most-specific method is ambiguous or no primary methods are
   applicable.

3. All applicable ``after()`` methods, from *least-specific* to most-specific,
   with methods at the same level of specificity executing in the reverse order
   from the order they were added.  (In other words, the more specific the
   ``after()`` condition, the "more after" it gets run!)

If any of these methods raises an uncaught exception, the overall function
execution terminates at that point, and methods later in the order are not
run.


"Around" Methods
----------------

Sometimes you need to recognize certain special cases, and perhaps not run
the entire generic function, or need to alter its return value in some way,
or perhaps trap and handle certain exceptions, etc.  You can do this with
"around" methods, which run "around" the entire "before/primary/after" sequence
described in the previous section.

A good way to think of this is that it's as if the "around" methods form a
separate generic function, whose default (least-specific) method is the
original, "inner" generic function.

When "around" methods are applicable on a given invocation of the generic
function, the most-specific "around" method is invoked.  It may then choose
to call its ``next_method`` to invoke the next-most-specific "around" method,
and so on.  When there are no more "around" methods, calling ``next_method``
instead invokes the "before", "primary", and "after" methods, according to
the sequence described in the previous section.  For example::

    >>> @around(BankAccount.withdraw, "amount > self.balance")
    ... def overdraft_fee(next_method,self,amount):
    ...     print "Adding overdraft fee of $25"
    ...     return next_method(self,amount+25)

    >>> acct.withdraw(20)
    Adding overdraft fee of $25
    Transferring 45 from overdraft protection


Custom Method Types
-------------------

If the standard before/after/around/when decorators don't work for your
application, you can create custom ones by defining your own "method types".

XXX

``peak.rules.implies()`` and ``peak.rules.overrides()`` are the generic
functions used to define implication relationships and method overriding, and
they are user-extensible.  There are two different engines available: one that
only handles type tuples, and one that supports arbitrary predicates.  Using
a string as a condition automatically upgrades a function's engine from one
type to the other.

XXX

Here's an example of a "pricing rules" generic function that accomodates tax
and discounts as well as upcharges.  (Don't worry if you don't understand it at
first glance; we'll go over the individual parts in detail later.)::

    >>> from peak.rules import Method, MethodList
    >>> from peak.rules import always_overrides, combine_actions

    >>> class DiscountMethod(Method):
    ...      """Subtract a discount"""
    ...
    ...      def override(self, other):
    ...          if self.__class__ == other.__class__:
    ...              return self.override(other.tail)  # drop the other one
    ...          return self.tail_with(combine_actions(self.tail, other))
    ...
    ...      def __call__(self, *args, **kw):
    ...          price = self.tail(*args, **kw)
    ...          return price - self.body(*args, **kw) * price
    
    >>> discount_when = DiscountMethod.make_decorator(
    ...     "discount_when", "Add the result of this calculation"
    ... )

    >>> class AddMethod(MethodList):
    ...     """Add the calculated values"""
    ...     def __call__(self, *args, **kw):
    ...         return sum(body(*args, **kw) for sig,prec,body in self.items)

    >>> add_when = AddMethod.make_decorator(
    ...     "add_when", "Add the result of this calculation"
    ... )

    >>> always_overrides(DiscountMethod, AddMethod)
    >>> always_overrides(AddMethod, Method)

The ``make_decorator()`` method of ``Method`` objects lets you create decorators
similar to ``when()`` et al.

XXX

We can now use these decorators to implement a generic function::

    >>> @abstract()
    ... def getPrice(product,customer=None,options=()):
    ...     """Get this product's price"""

    >>> class Product:
    ...     @add_when(getPrice)
    ...     def __addBasePrice(product,customer,options):
    ...         """Always include the product's base price"""
    ...         return product.base_price

    >>> shoes = Product()
    >>> shoes.base_price = 42

    >>> getPrice(shoes)
    42

And then we can create some pricing rules::

    >>> @add_when(getPrice, "'blue suede' in options")
    ... def blueSuedeUpcharge(product,customer,options):
    ...     return 24
    ...

    >>> @discount_when(getPrice, 
    ...    "customer=='Elvis' and 'blue suede' in options and product is shoes"
    ... )
    ... def ElvisGetsTenPercentOff(product,customer,options):
    ...     return .1

    >>> @add_when(getPrice)
    ... def everything_else_is_free(product, customer, options):
    ...     return 0

And try them out::

    >>> getPrice("something")
    0
    >>> getPrice(shoes)
    42
    >>> getPrice(shoes, options=['blue suede'])
    66
    >>> print getPrice(shoes, 'Elvis',options=['blue suede'])
    59.4
    >>> getPrice(shoes, 'Elvis')     # no suede, no discount!
    42


Porting Code from RuleDispatch
==============================

The major design differences between PEAK-Rules and RuleDispatch are:

1. It's designed for extensibility/pluggability from the ground up

2. It's built from the ground up using generic functions instead of adaptation,
   so its code is a lot more straightforward.  (The current implementation,
   combined with all its dependencies, is roughly the same number of lines as
   RuleDispatch *without* any of its dependencies -- and already has features
   that can't even be *added* to RuleDispatch.)

3. It generates custom bytecode for each generic function, to minimize calling
   and interpreter overhead, and to potentially allow compatibility with Psyco
   and PyPy in the future.  (Currently, neither Psyco nor PyPy support the
   "computed jump" trick used in the generated code, so don't try to
   Psyco-optimize any generic functions yet - it'll probably core dump!)

Because of its exensible design, PEAK-Rules can use custom-tuned engines for
specific application scenarios, and over time it may evolve the ability
to accept "tuning hints" to adjust the indexing techniques for special cases.

PEAK-Rules also supports the full method combination semantics of RuleDispatch
using a new decentralized approach, that allows you to easily create new method
types or combination semantics, complete with their own decorators (like
``when``, ``around``, etc.)

These decorators also all work with *existing* functions; you do not have to
predeclare a function generic in order to use it.  You can also omit the
condition from the decorator call, in which case the effect is the same as
RuleDispatch's ``strategy.default``, i.e. there is no condition.  Thus, you
can actually use PEAK-Rules's ``around()`` as a quick way to monkeypatch
existing functions, even ones defined by other packages.  (And the decorators
use the ``DecoratorTools`` package, so you can omit the ``@`` signs for
Python 2.3 compatibility.)

RuleDispatch was always conceived as a single implementation of a single
dispatch algorithm intended to be "good enough" for all uses.  Guido's argument
on the Py3K mailing list, however, was that applications with custom dispatch
needs should write custom dispatchers.  And I almost agree -- except that I
think they should get a RuleDispatch-like dispatcher for free, and be able to
tune or write ones to plug in for specialized needs.

The kicker was that Guido's experiment with type-tuple caching (a predecessor
algorithm to the Chambers-and-Chen algorithm used by RuleDispatch) showed it to
be fast *enough* for common uses, even without any C code, as long as you were
willing to do a little code generation.  The code was super-small, simple, and
fast enough that it got me thinking it was good enough for maybe 50% of what
you need generic functions for, especially if you added method combination.

And thus, PEAK-Rules was born, and RuleDispatch doomed to obsolescence.  (It
didn't help that RuleDispatch was a hurriedly-thrown-together experiment, with
poor testing and little documentation, either.)

So, if you are currently using RuleDispatch, we strongly advise that you port
your code.  To convert the most common RuleDispatch usages, simply do the
following:

* Replace ``@dispatch.on()`` and ``@dispatch.generic()`` with ``@abstract()``

* Replace ``@func.when(sig)`` with ``@when(func, sig)`` (and the same for
  ``before``, ``after``, and ``around``)

* When replacing ``@func.when(type)`` calls where ``func`` was defined with
  ``@dispatch.on``, use ``@func.when("isinstance(arg, type)")``, where ``arg``
  is the argument that was named in the ``@dispatch.on()`` call.


RuleDispatch Emulation
----------------------

If your code doesn't use much of the RuleDispatch API, you may be able to use
PEAK-Rules' "emulation API", which supports the following RuleDispatch APIs:

* ``dispatch.on``, ``dispatch.generic``, and dispatch.as``

* ``strategy.default``, ``strategy.Min``, ``strategy.Max``

* ``DispatchError``, ``NoApplicableMethods``, and ``AmbiguousMethod`` errors

* The ``when()``, ``before()``, ``after()`` and ``around()`` methods of generic
  functions.

(Note that some APIs may issue deprecation warnings (e.g. ``dispatch.as``), and
over time, the entire API will be deprecated.  Please update your code as soon
as practical.)

The emulation API does NOT support:

* custom combiners (use custom method types instead)

* The ``addMethod`` or ``__setitem__`` APIs for adding rules

* the ``clone()`` method of generics created with ``dispatch.on``

* PyProtocols (i.e., interfaces cannot be used for dispatching)

In the future, a PyProtocols emulation API may be added, but it doesn't exist
yet.

To use the emulation API, simply import ``dispatch`` from ``peak.rules``::

    >>> from peak.rules import dispatch

    >>> @dispatch.generic()     # roughly equivalent to @abstract()
    ... def a_function(an_arg, other_arg):
    ...     """Blah"""

    >>> @a_function.when((int, str))
    ... def a_when_int_str(an_arg, other_arg):
    ...     print "int and str"

    >>> a_function(42, "blue")
    int and str

    >>> a_function("blue", 42)
    Traceback (most recent call last):
      ...
    NoApplicableMethods: (('blue', 42), {})
    
Whether you use ``dispatch.generic`` or ``dispatch.on`` to define a generic
function, you can begin using ``peak.rules.when`` to declare methods
immediately::

    >>> @when(a_function, (str, int))
    ... def a_when_str_int(an_arg, other_arg):
    ...     print "str and int"

    >>> a_function("blue", 42)
    str and int

This means that you don't have to update your entire codebase at once; you can
port your method definitions incrementally, if desired.


------------
Mailing List
------------

Please direct questions regarding this package to the PEAK mailing list; see
http://www.eby-sarna.com/mailman/listinfo/PEAK/ for details.
