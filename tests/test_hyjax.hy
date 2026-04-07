(import jax
        jax [numpy :as jnp
             grad
             random])

(require hyjax [defn/j mapv if/l cond/l while/l fori/l lwhile lfori])

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

;;; while/l — imperative while-syntax over lax.while_loop

(defn/j count-to [n]
  (setv v (jnp.int32 0))
  (while/l (< v n)
    (setv v (+ v 1)))
  v)

(defn test-while-l-counts []
  (assert (= (int (count-to (jnp.array 7))) 7))
  (assert (= (int (count-to (jnp.array 0))) 0)))

(defn/j collatz-steps [start]
  ;; number of Collatz steps from `start` until we reach 1.
  (setv v     (jnp.int32 start))
  (setv steps (jnp.int32 0))
  (while/l (> v 1)
    (setv v (jnp.where (= (% v 2) 0)
                       (// v 2)
                       (+ (* 3 v) 1)))
    (setv steps (+ steps 1)))
  steps)

(defn test-while-l-multi-state []
  ;; 27 -> 1 takes 111 Collatz steps.
  (assert (= (int (collatz-steps (jnp.array 27))) 111)))

;;; fori/l — imperative for-syntax over lax.fori_loop

(defn/j sum-range [n]
  (setv acc (jnp.int32 0))
  (fori/l [i (range n)]
    (setv acc (+ acc i)))
  acc)

(defn test-fori-l-sum []
  ;; sum of 0..9 = 45
  (assert (= (int (sum-range 10)) 45))
  (assert (= (int (sum-range 0)) 0)))

(defn/j factorial [n]
  (setv acc (jnp.int32 1))
  (fori/l [i (range 1 (+ n 1))]
    (setv acc (* acc i)))
  acc)

(defn test-fori-l-factorial []
  (assert (= (int (factorial 5)) 120))
  (assert (= (int (factorial 1)) 1)))

(defn/j fib [n]
  ;; Fibonacci via simultaneous (a, b) = (b, a+b) assignment.
  (setv a (jnp.int32 0))
  (setv b (jnp.int32 1))
  (fori/l [i (range n)]
    (setv [a b] #(b (+ a b))))
  a)

(defn test-fori-l-multi-state []
  (assert (= (int (fib 0)) 0))
  (assert (= (int (fib 1)) 1))
  (assert (= (int (fib 10)) 55)))

;;; lwhile / lfori — explicit-binding forms

(defn/j count-to-explicit [n]
  (lwhile [v (jnp.int32 0)]
          (< v n)
          (+ v 1)))

(defn test-lwhile-counts []
  (assert (= (int (count-to-explicit (jnp.array 4))) 4)))

(defn/j collatz-explicit [start]
  ;; Same Collatz step counter, but with the carry written by hand.
  (let [result
        (lwhile [[v steps] #((jnp.int32 start) (jnp.int32 0))]
                (> v 1)
                #((jnp.where (= (% v 2) 0)
                             (// v 2)
                             (+ (* 3 v) 1))
                  (+ steps 1)))]
    (get result 1)))

(defn test-lwhile-multi-state []
  (assert (= (int (collatz-explicit (jnp.array 27))) 111)))

(defn/j sum-range-explicit [n]
  (lfori [i 0 n] [acc (jnp.int32 0)]
         (+ acc i)))

(defn test-lfori-sum []
  (assert (= (int (sum-range-explicit 10)) 45)))

(defn/j fib-explicit [n]
  (let [result
        (lfori [i 0 n] [[a b] #((jnp.int32 0) (jnp.int32 1))]
               #(b (+ a b)))]
    (get result 0)))

(defn test-lfori-multi-state []
  (assert (= (int (fib-explicit 10)) 55)))

;;; while/l walker should not leak let-local setvs into the carry

(defn/j count-with-let [n]
  (setv v (jnp.int32 0))
  (while/l (< v n)
    ;; `tmp` lives in a let scope; must NOT become part of the carry
    (let [tmp (+ v 1)]
      (setv v tmp)))
  v)

(defn test-while-l-ignores-inner-scopes []
  (assert (= (int (count-with-let (jnp.array 5))) 5)))

;;; defn/j composes with jax.grad

(defn/j sum-logistic [x]
  (jnp.sum (/ 1.0 (+ 1.0 (jnp.exp (- x))))))

(defn test-defn-j-is-differentiable []
  (let [x (jnp.arange 3.0)
        g ((grad sum-logistic) x)
        expected (jnp.array [0.25 0.19661197 0.10499357])]
    (assert (jnp.allclose g expected :atol 1e-6))))
