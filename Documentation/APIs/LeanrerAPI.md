
# Discussion on Learner and Trainer APIs




First let us lay out the notations and popular SGD algorithms in the literature. Let $l(w_t, x_i)$ be the loss function of sample $x_i$ at parameter weights $w_t$. In the literature, the loss are estimated with empirical loss over a data set $X$: 
$$
L(w_t; X) = \frac{1}{|X|}\sum_{x_i \in X} l(w_t, x_i)
$$
To parallelize the computation, we usually make use of its addictive structure and move $\frac{1}{|X|}$ inside the summation:
$$
L(w_t; X) = \sum_{x_i \in X} \frac{1}{|X|} l(w_t, x_i)
$$
**Note that the difference between the formulas in the literature and that are used in the computation usually causes confusion over the configuration of various optimization algorithms' parameters.**

## Stochastic gradient descent and its variance:

* Stochastic SGD, let $\eta$ be the learning rate, randomly sample $x_i$ from data set $X$:   

$$w_t = w_{t-1} -  \nabla_{w_t} \eta l(w_t, x_i)$$ 

* Stochastic minibatch SGD, randomaly sample a minibatch $B_j \subseteq X$:

$$w_t = w_{t-1} - \sum_{x_i \in B_j} \eta  \frac{1}{|B_j|} \nabla_{w_t} l(w_t, x_i)$$ 

* Stochastic momentent SGD: Let $\gamma$ be the momentum rate

    In the literature, $$
\begin{align}
u_t &= \gamma u_{t-1} + \sum_{x_i \in B_j} \frac{1}{|B_j|}\nabla_{w_t} l(w_t, x_i) \\
w_t &= w_{t-1} - \eta u_t
\end{align}
$$ 
Let $\gamma' = \gamma  \eta$, we have the formulas in a more addictively computable form:
$$
\begin{align}
v_t &= \gamma' v_{t-1} + \sum_{x_i \in B_j} \eta \frac{1}{|B_j|}\nabla_{w_t} l(w_t, x_i) \\
w_t &= w_{t-1} -  v_t
\end{align}
$$ 

* ADAM (and other stochastic SGDs with both momentum and 2nd momentum  involved): 

    Let $g_t = \sum_{x_i \in B} \frac{1}{|B|} \nabla_{w_t} l(w_t, x_i)$, and $g^2_t = g_t \odot g_t$ ($\odot$ i element-wise product operator of vectors)
    $$\begin{align}
    m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
    v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g^2_t\\
    \hat{m_t} &= m_t / (1 - \beta^t_1)\\
    \hat{v_t} &= v_t / (1 - \beta^t_2)\\
    w_{t} & = w_{t-1} - \eta \hat{m}_t \oslash (\hat{v}_t^{\frac{1}{2}} + \epsilon) 
    \end{align}$$ where $\oslash$ is element-wise division and $\hat{v}_t^{\frac{1}{2}}$ is element-wise square root operator. To distribute the batch dependent term $\frac{1}{|B|}$ into individual sample $x_i$, we need to rewrite the above equations into:
	$$\begin{align}
	m_t & = \beta_1 m_{t-1} + \sum_{x_i \in B} (1 - \beta_1)\frac{1}{|B|} \nabla_{w_t} l(w_t, x_i)\\
	v_t &= \beta_2 v_{t-1} + \sum_{x_i \in B} (1 - \beta_2)  \frac{1}{|B|^2} [\nabla_{w_t} l(w_t, x_i) \odot \nabla_{w_t} l(w_t, x_i)] \\
	\hat{m_t} &= m_t / (1 - \beta^t_1)\\
	\hat{v_t} &= v_t / (1 - \beta^t_2)\\
	w_{t} & = w_{t-1} - \eta \hat{m}_t \oslash (\hat{v}_t^{\frac{1}{2}} + \epsilon) 
	\end{align}$$



**TODO: 1) fill in equations for CNTK distributed training and their corresonding equations in the literature; 2) fill in equations for other implemented algorithms and their corresonding equations in the literature. **

## Regarding regularization terms
For example, if L2-regularization is applied, the regularized loss function: 
$$
 ll(w_t, x) = \frac{\lambda}{2}||w_t||^2 + l(w_t, x)
$$
Its gradient is 
$$
\nabla_{w_t}ll(w_t, x) = \lambda_{w_t} w_t + \nabla_{w_t} l(w_t, x)
$$
* The corresonding SGD update is 
    $$w_t = w_{t-1} - \eta \lambda w_t - \nabla_{w_t} \eta l(w_t, x_i)$$ 
* The corresponiding minibatch SGD update is:
$$
\begin{align}
    w_t &= w_{t-1}  - \sum_{x_i \in B_j}  (\frac{1}{|B_j|} \eta) \lambda w_t - \sum_{x_i \in B_j} (\frac{1}{|B_j|} \eta)   \nabla_{w_t} l(w_t, x_i) \\
        &= w_{t-1}  -   \lambda w_t - \sum_{x_i \in B_j}\frac{1}{|B_j|} \eta   \nabla_{w_t} l(w_t, x_i) 
\end{align}
$$

**We need to be careful about how the learning rate and minibatch size is handled in the update. From the above, either way will work but don't mix them.** 


# The API reqirement for learners
 
Key observations:
1. Various additional terms (e.g. inverse of sample size $\frac{1}{|B|}$, $\frac{1}{|B|^2}$, learning rate $\eta$,  firt momentum rate dependent term $1 - \beta_1$, or 2nd momentum rate dependent term $1 -\beta_2$) are multiplied into the gradient of an individual sample for the subsequent aggregation steps (e.g. summation) in a single GPU or across GPUs. 
2. The gradient of invidual example $\nabla_{w_t} l(w_t, x_i)$ might need to be transformed locally for individual sample, e.g. square operator: $\nabla_{w_t} l(w_t, x_i) \odot \nabla_{w_t} l(w_t, x_i)$.

**Please correct me if the above observation is incorrect or incomplete.**

Therefore, for various SGD algorithms, the key APIs are to 
1. Compute gradient for individual sample loss $\nabla l(w_t, x_i)$
2. Retrieve optimiation context for an indivdual sample. The optimization context might include the size of the minibatch which this example belongs to, learning rate, optimization algorithm specific rate such as $\beta_1$ and $\beta_2$
3. Subsequent one or more aggregation steps, e.g. summation in SGD. 


## The learning parameter should be functional
Each parameter can either be 
* a simple primitive value, e.g. int, double, string
* a functional which returns value based on the learning context --- compute the value on-the-fly based on other values in the context, or return a cached value.
These parameters are then embeded into the LearningContext.

For example, 
* In Python:
```python
    options = C.Learners.Options()
    options[ParameterName] = value 
    ...
    options['MomentumRate'] = lambda learning_context: learning_context['LearningRate'] **  learning_context['NumIteration'] 
```
* In C++:
```c++
    options = new CNTK::LearningOptions();
    options.Set(ParameterName, value);
    ...
    options.Set("MomentumRate", [](LearningContext& learning_context){
        return pow(learning_context.Get<double>("LearningRate"),  learning_context.Get<double>("NumIteration"));
    });    
```



# C++ LearnerContext and LearningOptions
The LearnerContext is encapsulated in the C++ class LearningContext. Its template member functions Get() and Set() support 

* Primitive types, e.g. int, double, 
* Object, 
* Functional object: ContextFunction<ReturnValueType>::function. A specific functional object is CNTK function which can handle batch and sequence axes for tensor operations. 

The Get() function returns the built-in context values: e.g. number of samples, number of sweeps and etc. These values are either computed on the fly or the training process explicitly call the Set() function to set it into the context for later reference. The context values are set in a lazy manner: If the value is not requested for the iteration, it won't be computed. 


```c++
/**
A class encapsulate learner context. 
**/
class LearningContext{
private: 
   ///Each supporting value type will has its own dictionary. This should be thread-safe map which will be implemented through mutex or through libraries which provide ConcurrentMap.
   template <typename T> GetDict();
public:
    typedef std:wstring KeyType;
    ///If T is non-functional return the values in the context; if T is functional, execute the functions. 
    template <typename T> const T& Get(const KeyType& key) const;
    template <typename T> const T& Set(const KeyType& key, const T& value) const;
};

/**
A template class which help to define functional object types.
*/
template <class ReturnValueType> ContextFunction{
public:
    typedef std:function<ReturnValueType(LearnerContext&)> function;
};

///LearningOptions have the same interface as LearningContext
class LearningOptions{
private: 
   ///Each supporting value type will has its own dictionary. This should be thread-safe map which will be implemented through mutex or through libraries which provide ConcurrentMap.
   template <typename T> GetDict();
public:
    typedef std:wstring KeyType;
    ///If T is non-functional return the values in the context; if T is functional, execute the functions. 
    template <typename T> const T& Get(const KeyType& key) const;
    template <typename T> const T& Set(const KeyType& key, const T& value) const;
};
```

### Additional data objects:
* RateOnMinibatch(double value, int as_for_num_of_samples)
* (to be completed)

### Schedules
* Schedules will be a function of the type ContextFunction<double>::function which retrieves the LearnerContext::Get(SWEEP_NUM) or LearnerContext::Get(ITERATION_NUM) and bases on the retreived value return the learning rates, momentum rates, minibatch size and other parameters.

### Metrics depending functions
* We will implement a list of basic built-in metrics in the context: loss, errrors and so on. This metrics will be computed in a lazy manner --- only when they are called, they will be computed for the corresponding iteration; and if they are requested, they will be computed only once per iteration.
* Customized metrics based on CNTK function defined through functional type.

# Example usages of the LearningContext
There should be a learning context per each iteration and per each worker. This context has various built-in values, e.g. number of sweeps, number of minibatches, and so on. These built-in values can have local values and global values which are referenced by different keys. 

To the make the idea concrete, examples are in Python are provided below (C++ counterparts are similar).

* For minibatch SGD:
```python
sample_coefficient =   learning_context[LEARNING_RATE] *  (1 / learning_context[MINIBATCH_SIZE])
w = w - sample_coefficient * gradient(sample)
```

* For ADAM:
```python
#The following will need to be converted into matrix notation
batch_size =  learning_context[MINIBATCH_SIZE]
alpha =  learning_context[ALPHA]
beta1 =  learning_context[BETA1]
beta2 = learning_context[BETA2]
sample_coefficient1 =  (1 - beta1)  *  (1 / batch_size)
sample_coefficient2 =  (1 - beta2) *  (1 / (batch_size * batch_size))
sample_gradient = gradient(sample)
mm = beta1 * m + sample_coefficient1 * sample_gradient #update the momentum 
vv = beta2 * v + sample_coefficient2 * sample_gradient #update the 2nd momentum
#...after all samples are collected into mm and vv
mm = mm / ( 1 - beta1 ** t) #this can be reformulated into a more distributed manner
vv = vv / (1 - beta2 ** t) #this can be reformulated into a more distributed manner
vv = sqrt(vv)
w = w - alpha * mm / (vv + epsilon)
#...synchronization
```


# Python learner interface

## SWIG tricks
* The C++ LearnerContext class will be ported into Python
* Python lambda functions will be translated into C++ function objects

## Python learner APIs with Options
* Each learner should be defined with the parameter as they are in the literature
    For example, 
    ```python
    adam_learner = C.learners.adam(alpha = 0.1, beta1 = 0.2, beta2 = 0.3)
    ```
* Each learner definition return partial definition of the learners through a lambda of type: Callable[[LearningOptions], C.learners.Learner]. It is the trainer or trainning session which finalizes the learner with additional option setting. A possible implementation is: 
    ```python
    def adam(alpha, beta1, beta2):
        def ret_adam(options: LearningOptions) ->  Learner:
            options['alpha'] = alpha
            options['beta1'] = beta1
            options['beta2'] = beta2
            return C.learners.sgd_impl(options)
        return ret_adam
    ```
* The learner can also be specified without any options so as to depend on the trainer or trainning sesssion specification. For example, 
    ```python
learner = C.learners.sgd  #  return type: Callable[[LearningOptions], Learner]
learner = C.learners.adam  #  return type: Callable[[LearningOptions], Learner]
learner = C.learners.adadelta  #  return type: Callable[[LearningOptions], Learner]
    ```
    
* Backward compatibility is achieved through setting property fields in options. For example,
    ```python
def sgd(parameters, lr, l1_regularization_weight=0, l2_regularization_weight=0, gaussian_noise_injection_std_dev=0, gradient_clipping_threshold_per_sample=np.inf, gradient_clipping_with_truncation=True):
    def ret_sgd(options: LearningOptions) ->  Learner:
        options['LearningRate'] = lr
        options['l1_regularization_weight'] = l1_regularization_weight
        ...
        return C.learners.sgd_impl(options)
    return ret_sgd
    ```



# New Trainer API
```python
def trainer(model, 
        loss, 
        learner, 
        options: C.learners.Options,
        # Based on the schedule, do something on the learning context: e.g. logging
        progress_callback: Callable[[Schedule, LearningContext]] 
        ) -> Trainer
```

## Backward compatibility is achieved through inspecting the python function parameter list and set the options correspondingly
For example, we also allow the following usage in Python but it won't be encouraged:
```python
trainer = C.learners.trainer(
       model = model,
       loss, # = cross_entropy(model, label)
       learner = learner, #C.learners.sgd
       reader = reader, #C.io.readers
       progress_logger, # 
       learning_rate, #   
       momentum_rate, #  e
       mb_size, # = lambda num_of_sample_processed: return adjusted_mb_size,
       # different learners and optimizers can register additional parmaters
       beta1 # for adam
       beta2 # for adam, 
       ....
)
```

# Regarding refactoring UnitType: per sample and per minibatch
* A helper class: RateOnMinibatch(value, as_for_number_of_samples, as_for_number_of_sequences)
    * value: The value of this rate.
    * as_for_number_of_samples: How many samples are this rate intended to apply on as a group in theory.
    * as_for_number_of_sequences: How many sequences are this rate intended to apply on as a group in theory.
    
Note that    
* Learning rate, momentum rate and second momentum rate will all use this class
* End user specifies the numbers as they are in the literature. The specific learners convert them into per sample coefficient for computational efficiency consideration.
    
Examples:    
```python
learner = C.learners.sgd(learning_rate = RateOnMinibatch(0.002, as_for_number_of_samples = 1))
learner = C.learners.sgd(learning_rate = RateOnMinibatch(0.2 ,as_for_number_of_samples = 1))
```

# Regarding schedules
As learning options are functional, the schedules can be specified in the following ways:
* In functional manner
```python
learner = C.learners.sgd(
    learning_rate = lambda learning_context: RateOnMinibatch(0.2 , 100) if learning_context['NumInteration'] < 100 else RateOnMinibatch(0.02, 100)
    )
```
* In array manner for backward compatibility
```python
learner = C.learners.sgd(
    learning_rate = [ (999, RateOnMinibatch(0.2 , 100)), (888, RateOnMinibatch(0.4 , 100))]
    )
```

# Other modules that are affected
* Readers: Reader should also be able to take functional minibatch size so as to feed the right size of minibath data  to the trainer/learner
* Checkpoints: Checkpoints need to store models, LearningContext and LearnerOptions correctly; and restore early version models with new options correctly.
