
"""
    create_transition_kernels(data, transition_kernel, prior)

creates a dictionary with observation times as keys and transition kernels as values. This assumes that the transition kernel only depends on the difference between observation times. Values are functions which take a state as argument and return a random state obtained through the transition kernel.


# Arguments
- `data::Dict{Real, Any}`: keys are observation times, values are observed data.
- `transition_kernel::Function`: a function for .


# Examples
```julia-repl
julia> bar([1, 2], [1, 2])
1
```
"""
function create_transition_kernels(data, transition_kernel, prior)
    times = data |> keys |> collect |> sort
    return zip(times, [prior; [transition_kernel(times[k]-times[k-1]) for k in 2:length(times)]]) |> Dict
end
