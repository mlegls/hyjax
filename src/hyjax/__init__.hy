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
