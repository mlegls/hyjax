# Hyjax

`pip install hyjax`

Hyjax is [Hy](https://github.com/hylang/hy) bindings for [JAX](https://github.com/google/jax). 


> When walking about the countryside of Italy, the people will not hesitate to tell you that JAX has [“una anima di pura programmazione funzionale”](https://www.sscardapane.it/iaml-backup/jax-intro/).
> –– <cite>[🔪 JAX - The Sharp Bits 🔪](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html?highlight=pure%20functional)</cite>

> “As one would expect from its goals, artificial intelligence research generates many significant programming problems. In other programming cultures this spate of problems spawns new languages. ... We toast the Lisp programmer who pens his thoughts within nests of parentheses.”
> -- <cite>Alan J. Perlis</cite>

> You now know enough to be dangerous with Hy. You may now smile villainously and sneak off to your Hydeaway to do unspeakable things.
> –– <cite>[Hy Tutorial](https://docs.hylang.org/en/stable/tutorial.html?highlight=hydeaway#next-steps)</cite>

The goals are
1. the name pun was funny
2. let fully JIT-compiled JAX code be written in idiomatic Lisp

Some examples loosely inspired by the [JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) tutorials are in `src/experiments.hy`. The [Hy](https://docs.hylang.org/en/stable/index.html) and [JAX](https://jax.readthedocs.io/en/latest/) docs might also be helpful.

## Features
- [x] `(mapv f vec)` as `vmap(f)(vec)`
- [x] `(if/ja pred then else)` as `lax.cond`
- [x] `(defn/j f [args] body)` as `@jit def f(args): ...`
  - [x] identical syntax to Hy's [`defn`](https://docs.hylang.org/en/stable/api.html#defn); supports other decorators, annotations, variadic & keyword args
- [x] `(if pred then else)` inside `defn/j` as `lax.cond`
  - [x] works with any macros that compile to `if`, such as `cond`
- [ ] binding for `lax.while_loop`
- [ ] binding for `lax.fori_loop`
