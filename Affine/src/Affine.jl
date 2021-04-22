module Affine

function length_transform(x::Real)
    return 128 / 100 * x - 64
end

function length_transform_inverse(x::Real)
    return 100 / 128 * x + 50
end

function D_transform(D::Real)
    return (128/100)^2 * D
end

function D_transform_inverse(D::Real)
    return (100/128)^2 * D
end


end # module
