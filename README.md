# Hyjax

```bash
uv add hyjax         # or: pip install hyjax
```

Dev setup:

```bash
git clone https://github.com/mlegls/hyjax
cd hyjax
uv sync
uv run pytest                       # run the test suite
uv run hy examples/experiments.hy   # run the example script
```

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

Some examples loosely inspired by the [JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) tutorials are in `examples/experiments.hy`. The [Hy](https://docs.hylang.org/en/stable/index.html) and [JAX](https://jax.readthedocs.io/en/latest/) docs might also be helpful.

## Features
- [x] `(defn/j f [args] body)` as `@jit def f(args): ...`
  - `/j` for **j**it; identical syntax to Hy's [`defn`](https://docs.hylang.org/en/stable/api.html#defn); supports other decorators, annotations, variadic & keyword args
- [x] `(mapv f vec)` as `vmap(f)(vec)`
- [x] `(if/l pred then else)` as `lax.cond`
  - `/l` for **l**ax; if-like binary form
- [x] `(cond/l pred1 result1 pred2 result2 ... default)` as a chain of `lax.cond`
  - multi-branch form, mirroring Lisp `cond`
- [x] `(lcond [bindings] pred then else)` — lower-level form with explicit closed-over bindings
- [x] `(while/l [v init] cond body)` as `lax.while_loop`
  - state pattern may destructure: `(while/l [[a b] #(0 1)] (< a 10) #((+ a 1) (* b 2)))`
- [x] `(fori/l [i lo hi] [acc init] body)` as `lax.fori_loop`
- [ ] binding for `lax.while_loop`
- [ ] binding for `lax.fori_loop`
