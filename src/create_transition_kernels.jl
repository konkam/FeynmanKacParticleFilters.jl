
"""
    create_transition_kernels(data, transition_kernel, prior)

Creates a dictionary with observation times as keys and transition kernels as values. This assumes that the transition kernel only depends on the difference between observation times. Values are functions which take a state as argument and return a random state obtained through the transition kernel.


# Arguments
- `data::Dict{Real, Any}`: keys are observation times, values are observed data.
- `transition_kernel::Function`: a function that takes time difference as argument and returns a transition kernel (a function which takes a state as argument and return a random state obtained through the transition kernel).
- `prior::Distribution`: a prior distribution that can be dispatched to rand().


# Examples
```julia-repl
julia> bar([1, 2], [1, 2])
1
```
"""
function create_transition_kernels(data, transition_kernel, prior)
    times = data |> keys |> collect |> sort
    prior_rng(x) = rand(prior)
    return zip(times, [prior_rng; [transition_kernel(times[k]-times[k-1]) for k in 2:length(times)]]) |> Dict
end
