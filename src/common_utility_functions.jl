# function bind_rows(dflist)
#     vcat(dflist...)
# end

"Normalises a vector"
function normalise(x)
    return x ./ sum(x)
end

function reverse_time_in_dict(dict)
    times = dict |> keys |> collect |> sort
    return Dict(zip(reverse(times), (dict[k] for k in times)))
end

function reverse_time_in_dict_except_first(dict)
    times = dict |> keys |> collect |> sort
    revtimes = reverse(times[2:end])
    return merge(Dict(times[1] => dict[times[1]]), Dict(zip(revtimes, (dict[t] for t in times[2:end]))))
end
