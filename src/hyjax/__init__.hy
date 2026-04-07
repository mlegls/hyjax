(import jax
        jax [lax]
        hy.pyops [reduce +]
        hy.models [List :as HyList
                   Expression
                   Symbol])



;;; Helper functions & macros

(setv _opaque-scope-heads
  ;; Heads whose bodies we don't walk at all: nested function/macro
  ;; definitions and comprehensions. Their internal setvs never mutate
  ;; carry state from the enclosing loop.
  #{"fn" "fn/a" "defn" "defn/a" "defn/j" "defmacro"
    "lfor" "sfor" "dfor" "gfor"})

(defn _pattern-names [target]
  "Yield the Symbols appearing as names in a (setv ...) / let target."
  (cond
    (isinstance target Symbol) [target]
    (isinstance target HyList) (lfor x target
                                     :if (isinstance x Symbol)
                                     x)
    True []))

(defn _collect-setv-targets [form]
  "Walk `form` and return the symbols that appear as (setv ...) targets
  and refer to bindings in the *enclosing* scope.

  Descends into `let` and `for` while tracking their local names so
  those aren't mis-classified as carry state. Fully skips nested
  fn/defn/comprehension bodies. Handles destructuring targets like
  (setv [a b] expr).
  "
  (let [targets []
        seen    (set)]
    (defn add! [sym]
      (let [s (str sym)]
        (when (not-in s seen)
          (.add seen s)
          (.append targets sym))))
    (defn visit [node locals]
      (when (and (isinstance node Expression) (> (len node) 0))
        (let [head (get node 0)
              head-name (if (isinstance head Symbol) (str head) None)]
          (cond
            ;; setv: collect targets not shadowed by a local, then walk values.
            (= head-name "setv")
            (do
              (for [i (range 1 (len node) 2)]
                (for [name (_pattern-names (get node i))]
                  (when (not-in (str name) locals)
                    (add! name))))
              (for [i (range 2 (len node) 2)]
                (visit (get node i) locals)))

            ;; let: bindings are [name expr name expr ...]; names become
            ;; local to the body, but value exprs see the outer locals.
            (= head-name "let")
            (let [bindings  (get node 1)
                  new-local (set locals)]
              (for [i (range 0 (len bindings) 2)]
                (for [name (_pattern-names (get bindings i))]
                  (.add new-local (str name))))
              (for [i (range 1 (len bindings) 2)]
                (visit (get bindings i) locals))
              (for [child (cut node 2 None)]
                (visit child new-local)))

            ;; for: (for [x iter ...] body); loop vars are local to body.
            (= head-name "for")
            (let [bindings  (get node 1)
                  new-local (set locals)]
              (for [i (range 0 (len bindings) 2)]
                (for [name (_pattern-names (get bindings i))]
                  (.add new-local (str name))))
              (for [i (range 1 (len bindings) 2)]
                (visit (get bindings i) locals))
              (for [child (cut node 2 None)]
                (visit child new-local)))

            ;; Opaque scopes: skip entirely.
            (in head-name _opaque-scope-heads)
            None

            True
            (for [child node] (visit child locals))))))
    (visit form (set))
    targets))

(defn find-symbols [expr]
  (match expr
    [] []
    (Expression [head #* tail]) (find-symbols tail) ; first element in Expression is special
    [head] (find-symbols head)
    [head #* tail] (+ (find-symbols head) (find-symbols tail))
    (Symbol) [expr]
    _ []))                       

(defn unique [lst] (list (set lst)))

(defn fn-with-bindings [bindings #* body]
  (let [binding-list (hy.gensym "binding-list")]
    `(fn [~binding-list]
       (let [~bindings ~binding-list]
         ~@body))))

(defn _lcond-form [bindings pred true-case false-case]
  "Build a (lax.cond ...) expression with an if-like surface syntax."
  `(hy.I.jax.lax.cond ~pred
                      ~(fn-with-bindings bindings true-case)
                      ~(fn-with-bindings bindings false-case)
                      ~bindings))

(defmacro lcond [bindings pred true-case false-case]
  "lax cond, with if-like syntax"
  (_lcond-form bindings pred true-case false-case))

(defn _cond/l [args]
  (if (> (len args) 1)
    `(if/l ~(get args 0)
      ~(get args 1)
      ~(_cond/l (cut args 2 None)))
    (get args 0)))

;;; API Macros

(defmacro if/l [pred true-case false-case]
  "if-like wrapper around lax.cond with auto binding detection"
  (let [bindings (unique (reduce + (map find-symbols [pred true-case false-case])))]
    (_lcond-form bindings pred true-case false-case)))

(defmacro cond/l [#* args]
  "cond-like chain of lax.cond branches; takes an odd number of forms
  (pred1 result1 pred2 result2 ... default)"
  (if (not (% (len args) 2))
    (raise (ValueError "cond/l requires an odd number of arguments"))
    (_cond/l args)))


(defmacro defn/j [#* args]
  "define jit-compiled function; replacing control flow with lax constructs"
  (match args
    ; plain defn
    [(| (Symbol name) (Expression name)) (HyList args) #* body]
    `(defn [hy.I.jax.jit] ~name ~args ~@body)

    ; with decorators
    [(HyList decorators) (| (Symbol name) (Expression name)) (HyList args) #* body]
    `(defn [hy.I.jax.jit ~@decorators] ~name ~args ~@body)))

(defmacro mapv [f v]
  `((hy.I.jax.vmap ~f) ~v))

(defmacro lwhile [binding cond-expr body-expr]
  "Explicit-binding lax.while_loop wrapper. Lower-level form of while/l.

  The state binding is [pattern init]; the body is a single expression
  that returns the new state. The pattern may destructure a tuple.

      (lwhile [v 0]
              (< v 10)
              (+ v 1))

      (lwhile [[a b] #(0 1)]
              (< a 10)
              #((+ a 1) (* b 2)))

  Use this when you want full control over what's in the carry, e.g.
  to keep loop-local temporaries out of it, or when the body is more
  naturally an expression than a sequence of (setv ...) statements.
  "
  (when (!= (len binding) 2)
    (raise (ValueError "lwhile binding must be [pattern init]")))
  (let [pattern (get binding 0)
        init    (get binding 1)
        carry   (hy.gensym "carry")]
    `(hy.I.jax.lax.while_loop
       (fn [~carry] (setv ~pattern ~carry) ~cond-expr)
       (fn [~carry] (setv ~pattern ~carry) ~body-expr)
       ~init)))

(defmacro lfori [index-binding state-binding body-expr]
  "Explicit-binding lax.fori_loop wrapper. Lower-level form of fori/l.

      (lfori [i 0 n]    ; index name, lower, upper
             [acc 0]    ; carried state
             (+ acc i))

  Like lwhile, the state pattern may destructure a tuple.
  "
  (when (!= (len index-binding) 3)
    (raise (ValueError "lfori index binding must be [i lower upper]")))
  (when (!= (len state-binding) 2)
    (raise (ValueError "lfori state binding must be [pattern init]")))
  (let [i       (get index-binding 0)
        lower   (get index-binding 1)
        upper   (get index-binding 2)
        pattern (get state-binding 0)
        init    (get state-binding 1)
        carry   (hy.gensym "carry")]
    `(hy.I.jax.lax.fori_loop
       ~lower ~upper
       (fn [~i ~carry] (setv ~pattern ~carry) ~body-expr)
       ~init)))

(defmacro while/l [cond-expr #* body]
  "lax.while_loop wrapper with natural while-like syntax.

  Variables assigned with (setv ...) inside the body are detected
  automatically and carried as the loop state. They must already be
  bound in the enclosing scope; their final values are written back
  after the loop exits.

      (setv v 0)
      (while/l (< v 10)
        (setv v (+ v 1)))
      ; v is now 10

  For simultaneous updates, use destructuring assignment:

      (setv a 0  b 1)
      (while/l (< a 100)
        (setv [a b] #(b (+ a b))))
  "
  (let [carry-syms (_collect-setv-targets `(do ~@body))
        carry      (hy.gensym "carry")]
    (when (= (len carry-syms) 0)
      (raise (ValueError
               "while/l body must assign to at least one variable via (setv ...)")))
    `(setv [~@carry-syms]
           (hy.I.jax.lax.while_loop
             (fn [~carry]
               (setv [~@carry-syms] ~carry)
               ~cond-expr)
             (fn [~carry]
               (setv [~@carry-syms] ~carry)
               ~@body
               [~@carry-syms])
             [~@carry-syms]))))

(defmacro fori/l [binding #* body]
  "lax.fori_loop wrapper with for-like syntax.

  The iterable must be a literal (range ...) form; (range n) iterates
  0..n and (range lo hi) iterates lo..hi. Variables assigned with
  (setv ...) inside the body become the carried state (same rules as
  while/l) and must be pre-bound in the enclosing scope.

      (setv acc 0)
      (fori/l [i (range n)]
        (setv acc (+ acc i)))
  "
  (when (!= (len binding) 2)
    (raise (ValueError "fori/l binding must be [i (range ...)]")))
  (let [i-sym      (get binding 0)
        range-expr (get binding 1)]
    (when (not (and (isinstance range-expr Expression)
                    (> (len range-expr) 0)
                    (isinstance (get range-expr 0) Symbol)
                    (= (str (get range-expr 0)) "range")))
      (raise (ValueError "fori/l iterable must be a (range ...) form")))
    (let [range-args (cut range-expr 1 None)
          carry-syms (_collect-setv-targets `(do ~@body))
          carry      (hy.gensym "carry")
          lower None
          upper None]
      (cond
        (= (len range-args) 1)
        (do (setv lower 0) (setv upper (get range-args 0)))
        (= (len range-args) 2)
        (do (setv lower (get range-args 0))
            (setv upper (get range-args 1)))
        True
        (raise (ValueError
                 "fori/l only supports (range upper) or (range lower upper)")))
      (when (= (len carry-syms) 0)
        (raise (ValueError
                 "fori/l body must assign to at least one variable via (setv ...)")))
      `(setv [~@carry-syms]
             (hy.I.jax.lax.fori_loop
               ~lower ~upper
               (fn [~i-sym ~carry]
                 (setv [~@carry-syms] ~carry)
                 ~@body
                 [~@carry-syms])
               [~@carry-syms])))))
