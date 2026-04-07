(import jax
        jax [lax]
        hy.pyops [reduce +]
        hy.models [List :as HyList
                   Expression
                   Symbol])



;;; Helper functions & macros

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

(defmacro if/lp [bindings pred true-case false-case]
  "Explicit-binding lax.cond wrapper. Lower-level form of if/l.

  The bindings list is the explicit closure passed through lax.cond;
  use this when you don't want auto-detection of free symbols.

      (if/lp [a b] (= a b) (+ a 1) (- b 2))
  "
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

(defmacro while/lp [binding cond-expr body-expr]
  "Explicit-binding lax.while_loop wrapper. Lower-level form of while/l.

  The state binding is [pattern init]; the body is a single expression
  that returns the new state. The pattern may destructure a tuple.

      (while/lp [v 0]
                (< v 10)
                (+ v 1))

      (while/lp [[a b] #(0 1)]
                (< a 10)
                #((+ a 1) (* b 2)))

  Use this when you want full control over what's in the carry, e.g.
  to keep loop-local temporaries out of it, or when the body is more
  naturally an expression than a sequence of (setv ...) statements.
  "
  (when (!= (len binding) 2)
    (raise (ValueError "while/lp binding must be [pattern init]")))
  (let [pattern (get binding 0)
        init    (get binding 1)
        carry   (hy.gensym "carry")]
    `(hy.I.jax.lax.while_loop
       (fn [~carry] (setv ~pattern ~carry) ~cond-expr)
       (fn [~carry] (setv ~pattern ~carry) ~body-expr)
       ~init)))

(defmacro fori/lp [index-binding state-binding body-expr]
  "Explicit-binding lax.fori_loop wrapper. Lower-level form of fori/l.

      (fori/lp [i 0 n]    ; index name, lower, upper
               [acc 0]    ; carried state
               (+ acc i))

  Like while/lp, the state pattern may destructure a tuple.
  "
  (when (!= (len index-binding) 3)
    (raise (ValueError "fori/lp index binding must be [i lower upper]")))
  (when (!= (len state-binding) 2)
    (raise (ValueError "fori/lp state binding must be [pattern init]")))
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

(defn _check-carry-names [carry ctx]
  (when (or (not (isinstance carry HyList)) (= (len carry) 0))
    (raise (ValueError (+ ctx " carry must be a non-empty [name ...] list"))))
  (for [n carry]
    (when (not (isinstance n Symbol))
      (raise (ValueError (+ ctx " carry names must be plain symbols"))))))

(defmacro while/l [carry cond-expr body-expr]
  "lax.while_loop wrapper with auto-init from the enclosing scope.

  `carry` is an explicit list of state variable *names*; each name
  must already be bound in the surrounding scope, and those values
  become the loop's initial state. `cond-expr` and `body-expr` are
  evaluated with those names destructured from the carry; `body-expr`
  is a single expression that evaluates to the new carry.

      (setv v 0)
      (setv v (while/l [v] (< v 10) (+ v 1)))   ; single-state

      (setv a 0  b 1)
      (setv [a b] (while/l [a b]                 ; multi-state
                           (< a 100)
                           #(b (+ a b))))

  With one carry name, the init and body are scalars; with N > 1,
  they are N-tuples in the declared order.
  "
  (_check-carry-names carry "while/l")
  (let [names (list carry)
        param (hy.gensym "carry")]
    (if (= (len names) 1)
      (let [n (get names 0)]
        `(hy.I.jax.lax.while_loop
           (fn [~param] (setv ~n ~param) ~cond-expr)
           (fn [~param] (setv ~n ~param) ~body-expr)
           ~n))
      `(hy.I.jax.lax.while_loop
         (fn [~param] (setv [~@names] ~param) ~cond-expr)
         (fn [~param] (setv [~@names] ~param) ~body-expr)
         #(~@names)))))

(defmacro fori/l [index-binding carry body-expr]
  "lax.fori_loop wrapper with auto-init from the enclosing scope.

  The iterable is a literal (range ...) form:
  (range n) iterates 0..n, (range lo hi) iterates lo..hi.
  `carry` is an explicit list of state variable names (see `while/l`).

      (setv acc 0)
      (setv acc (fori/l [i (range n)] [acc] (+ acc i)))

      (setv a 0  b 1)
      (setv [a b] (fori/l [i (range n)] [a b] #(b (+ a b))))
  "
  (when (!= (len index-binding) 2)
    (raise (ValueError "fori/l index binding must be [i (range ...)]")))
  (_check-carry-names carry "fori/l")
  (let [i-sym      (get index-binding 0)
        range-expr (get index-binding 1)]
    (when (not (and (isinstance range-expr Expression)
                    (> (len range-expr) 0)
                    (isinstance (get range-expr 0) Symbol)
                    (= (str (get range-expr 0)) "range")))
      (raise (ValueError "fori/l iterable must be a (range ...) form")))
    (let [range-args (cut range-expr 1 None)
          names (list carry)
          param (hy.gensym "carry")
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
      (if (= (len names) 1)
        (let [n (get names 0)]
          `(hy.I.jax.lax.fori_loop
             ~lower ~upper
             (fn [~i-sym ~param] (setv ~n ~param) ~body-expr)
             ~n))
        `(hy.I.jax.lax.fori_loop
           ~lower ~upper
           (fn [~i-sym ~param] (setv [~@names] ~param) ~body-expr)
           #(~@names))))))
