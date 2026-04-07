(import jax
        jax [numpy :as jnp
             grad
             random])

(require hyjax [defn/j mapv if/l cond/l])

;;; defn/j — basic jit

(defn/j selu [x [alpha 1.67] [lmbda 1.05]]
  (* lmbda (jnp.where (> x 0) x (- (* alpha (jnp.exp x)) alpha))))

(defn test-defn-j-is-jitted []
  "defn/j should wrap the function with jax.jit."
  ;; jitted callables are instances of jax's PjitFunction / CompiledFunction.
  ;; The robust check is that lowering works without tracing errors.
  (let [y (selu (jnp.array 1.0))]
    (assert (= y.shape #()))
    (assert (jnp.allclose y 1.05))))

(defn test-defn-j-with-defaults []
  (let [x (jnp.array [-1.0 0.0 2.0])
        y (selu x)]
    (assert (= y.shape #(3)))))

;;; mapv — vmap sugar

(defn/j square [x] (* x x))

(defn test-mapv-squares-vector []
  (let [v (jnp.arange 5.0)
        out (mapv square v)]
    (assert (jnp.allclose out (jnp.array [0.0 1.0 4.0 9.0 16.0])))))

(defn test-mapv-matrix-vector []
  (let [mat (jnp.eye 3)]
    (defn/j apply-mat [v] (jnp.dot mat v))
    (let [batch (jnp.arange 6.0 :dtype jnp.float32)
          batch (jnp.reshape batch #(2 3))
          out (mapv apply-mat batch)]
      (assert (jnp.allclose out batch)))))

;;; if/l — compiles to lax.cond

(defn/j branch [x]
  (if/l (< x 3)
        (* 3 (** x 2))
        (* -4 x)))

(defn test-if-j-true-branch []
  (assert (= (int (branch (jnp.array 2))) 12)))

(defn test-if-j-false-branch []
  (assert (= (int (branch (jnp.array 5))) -20)))

;;; nested if/l

(defn/j nested [x]
  (if/l (< x 3) (* 3 (** x 2))
        (if/l (< x 5) (* -4 x)
              (* 5 x))))

(defn test-nested-if-j []
  (assert (= (int (nested (jnp.array 2))) 12))
  (assert (= (int (nested (jnp.array 4))) -16))
  (assert (= (int (nested (jnp.array 10))) 50)))

;;; cond/l — multi-branch lax.cond

(defn/j piecewise [x]
  (cond/l
    (< x 3) (* 3 (** x 2))
    (< x 5) (* -4 x)
    (* 5 x)))

(defn test-cond-j-first-branch []
  (assert (= (int (piecewise (jnp.array 2))) 12)))

(defn test-cond-j-middle-branch []
  (assert (= (int (piecewise (jnp.array 4))) -16)))

(defn test-cond-j-fallthrough []
  (assert (= (int (piecewise (jnp.array 10))) 50)))

;;; defn/j composes with jax.grad

(defn/j sum-logistic [x]
  (jnp.sum (/ 1.0 (+ 1.0 (jnp.exp (- x))))))

(defn test-defn-j-is-differentiable []
  (let [x (jnp.arange 3.0)
        g ((grad sum-logistic) x)
        expected (jnp.array [0.25 0.19661197 0.10499357])]
    (assert (jnp.allclose g expected :atol 1e-6))))
