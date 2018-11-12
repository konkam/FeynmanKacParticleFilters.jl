function bind_rows(dflist)
    vcat(dflist...)
end

"Normalises a vector"
function normalise(x)
    return x/sum(x)
end
