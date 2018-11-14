function create_potential_functions(data, potential)
    times = data |> keys |> collect |> sort
    return zip(times, [potential(data[t]) for t in times]) |> Dict
end
