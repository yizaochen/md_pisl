module Affine

function length_transform(x::Real)
    return 128 / 100 * x - 64
end

function length_transform(x_left_ps::Real, x_right_ps::Real, x_left_r::Real, x_right_r::Real, x::Real)
    factor = (x_right_r - x_left_r) / (x_right_ps - x_left_ps)
    middle_r_old = (factor * x_right_ps + factor * x_left_ps) / 2
    middle_r_new = (x_right_r + x_left_r) / 2
    return factor * x + (middle_r_new - middle_r_old)
end

function length_transform_inverse(x::Real)
    return 100 / 128 * x + 50
end

function length_transform_inverse(x_left_ps::Real, x_right_ps::Real, x_left_r::Real, x_right_r::Real, x::Real)
    factor = (x_right_ps - x_left_ps) / (x_right_r - x_left_r)
    middle_ps_old = (factor * x_right_r + factor * x_left_r) / 2
    middle_ps_new = (x_right_ps + x_left_ps) / 2
    return factor * x + (middle_ps_new - middle_ps_old)
end

function time_transform(t::Real)
    return 1e-2 * t
end

function time_transform(factor::Real, t::Real)
    return factor * t
end

function time_transform_inverse(t::Real)
    return 1e2 * t
end

function time_transform_inverse(factor::Real, t::Real)
    return (1 / factor) * t
end

function D_transform(D::Real)
    return (128/100)^2 *  1e2 * D
end

function D_transform(x_left_ps::Real, x_right_ps::Real, x_left_r::Real, x_right_r::Real, time_factor::Real, D::Real)
    length_factor = (x_right_r - x_left_r) / (x_right_ps - x_left_ps)
    return (length_factor)^2 *  (1 / time_factor) * D
end

function D_transform_inverse(D::Real)
    return (100/128)^2 * 1e-2 * D
end

function D_transform_inverse(x_left_ps::Real, x_right_ps::Real, x_left_r::Real, x_right_r::Real, time_factor::Real, D::Real)
    length_factor = (x_right_ps - x_left_ps) / (x_right_r - x_left_r)
    return (length_factor)^2 * time_factor * D
end


end # module
