function sigmoid_eta(x_center::Real, x::Real, scale_factor::Real, translation_factor::Real)
    z = x - x_center
    return (one(z) / (one(z) + exp(-z))) * scale_factor + translation_factor
end

function linear_eta(xavg::Real, xratio::Real, yleft::Real, yright::Real, x::Real, c::Real)
    """
    model: y = mx + c
    """
    xright = xavg + xratio
    xleft = xavg - xratio
    m = (yright - yleft) / (xright - xleft)
    return m * x + c
end