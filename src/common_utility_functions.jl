# function bind_rows(dflist)
#     vcat(dflist...)
# end

"Normalises a vector"
function normalise(x)
    return x/sum(x)
end

function reverse_time_in_dict(dict)
    times = dict |> keys |> collect |> sort
    return Dict(zip(reverse(times), (dict[k] for k in times)))
end
