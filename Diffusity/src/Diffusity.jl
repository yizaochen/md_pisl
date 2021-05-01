module Diffusity

function sigmoid(x_center::Real, x::Real, scale_factor::Real, translation_factor::Real)
    z = x - x_center
    return (one(z) / (one(z) + exp(-z))) * scale_factor + translation_factor
end

end # module
