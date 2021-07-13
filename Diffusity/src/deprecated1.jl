function get_Bij(i::Int64, j::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    return alpha_t0[i] * psi_photon_psi[i,j]
end

function get_Bj_array(j::Int64, Nv::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    return [get_Bij(i, j, alpha_t0, psi_photon_psi) for i=1:Nv]
end

function get_expDLambda0Deltat_array(D::Real, lambda0_array::Array{Float64,1}, deltat::Real)
    DLambdaDeltat_array = (D * deltat) * lambda0_array
    return exp.(-DLambdaDeltat_array)
end

function get_fjD(D::Real, lambda0_array::Array{Float64,1}, deltat::Real, j::Int64, Nv::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    Bj_array = get_Bj_array(j, Nv, alpha_t0, psi_photon_psi)
    exp_array = get_expDLambda0Deltat_array(D, lambda0_array, deltat)
    return dot(exp_array, Bj_array)
end

function get_fjD_derivative(D::Real, lambda0_array::Array{Float64,1}, deltat::Real, j::Int64, Nv::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    Bj_array = get_Bj_array(j, Nv, alpha_t0, psi_photon_psi)
    exp_array = get_expDLambda0Deltat_array(D, lambda0_array, deltat)
    lambda0Deltat_exp_array = (deltat * lambda0_array) .* exp_array
    return dot(-lambda0Deltat_exp_array, Bj_array)
end

function get_l_by_fjD(D::Real, lambda0_array::Array{Float64,1}, deltat::Real, Nv::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    return log(sum(get_fjD(D, lambda0_array, deltat, j, Nv, alpha_t0, psi_photon_psi)^2 for j=1:Nv))
end

function get_fjD_square_derivative(D::Real, lambda0_array::Array{Float64,1}, deltat::Real, j::Int64, Nv::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    fjD = get_fjD(D, lambda0_array, deltat, j, Nv, alpha_t0, psi_photon_psi)
    fjD_derivative = get_fjD_derivative(D, lambda0_array, deltat, j, Nv, alpha_t0, psi_photon_psi)
    return 2 * fjD * fjD_derivative
end

function get_py_derivative(D::Real, lambda0_array::Array{Float64,1}, deltat::Real, Nv::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    return sum(get_fjD_square_derivative(D, lambda0_array, deltat, j, Nv, alpha_t0, psi_photon_psi) for j=1:Nv)
end

function get_l_derivative(D::Real, lambda0_array::Array{Float64,1}, deltat::Real, Nv::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    py = sum(get_fjD(D, lambda0_array, deltat, j, Nv, alpha_t0, psi_photon_psi)^2 for j=1:Nv)
    py_derivative = get_py_derivative(D, lambda0_array, deltat, Nv, alpha_t0, psi_photon_psi)
    return py_derivative / py 
end

### Taylor Expansion Part
function get_aj(j::Int64, Nv::Int64, lambda0_array::Array{Float64,1}, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    return dot(lambda0_array.^2, get_Bj_array(j, Nv, alpha_t0, psi_photon_psi))
end

function get_bj(j::Int64, Nv::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    return sum(get_Bj_array(j, Nv, alpha_t0, psi_photon_psi))
end

function get_cj(j::Int64, Nv::Int64, lambda0_array::Array{Float64,1}, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    return dot(lambda0_array, get_Bj_array(j, Nv, alpha_t0, psi_photon_psi))
end

#### Taylor v0: all terms include
function single_term_taylor_v0(D::Real, deltat::Real, j::Int64, Nv::Int64, lambda0_array::Array{Float64,1}, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    aj = get_aj(j, Nv, lambda0_array, alpha_t0, psi_photon_psi)
    bj = get_bj(j, Nv, alpha_t0, psi_photon_psi)
    cj = get_cj(j, Nv, lambda0_array, alpha_t0, psi_photon_psi)
    first_term = - deltat * bj * cj
    second_term = D * deltat^2 * (aj*bj - cj^2)
    third_term = D^2 * deltat^3 * aj * cj
    return first_term + second_term + third_term
end

function get_py_derivative_taylor_v0(D::Real, lambda0_array::Array{Float64,1}, deltat::Real, Nv::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    return 2 * sum(single_term_taylor_v0(D, deltat, j, Nv, lambda0_array, alpha_t0, psi_photon_psi) for j=1:Nv)
end

function get_l_derivative_taylor_v0(D::Real, lambda0_array::Array{Float64,1}, deltat::Real, Nv::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    py = sum(get_fjD(D, lambda0_array, deltat, j, Nv, alpha_t0, psi_photon_psi)^2 for j=1:Nv)
    py_derivative = get_py_derivative_taylor_v0(D, lambda0_array, deltat, Nv, alpha_t0, psi_photon_psi)
    return py_derivative / py 
end

#### Taylor v1: first_term + second_term
function single_term_taylor_v1(D::Real, deltat::Real, j::Int64, Nv::Int64, lambda0_array::Array{Float64,1}, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    aj = get_aj(j, Nv, lambda0_array, alpha_t0, psi_photon_psi)
    bj = get_bj(j, Nv, alpha_t0, psi_photon_psi)
    cj = get_cj(j, Nv, lambda0_array, alpha_t0, psi_photon_psi)
    first_term = - deltat * bj * cj
    second_term = D * deltat^2 * (aj*bj - cj^2)
    return first_term + second_term
end

function get_py_derivative_taylor_v1(D::Real, lambda0_array::Array{Float64,1}, deltat::Real, Nv::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    return 2 * sum(single_term_taylor_v1(D, deltat, j, Nv, lambda0_array, alpha_t0, psi_photon_psi) for j=1:Nv)
end

function get_l_derivative_taylor_v1(D::Real, lambda0_array::Array{Float64,1}, deltat::Real, Nv::Int64, alpha_t0::Array{Float64,2}, psi_photon_psi::Array{Float64,2})
    py = sum(get_fjD(D, lambda0_array, deltat, j, Nv, alpha_t0, psi_photon_psi)^2 for j=1:Nv)
    py_derivative = get_py_derivative_taylor_v1(D, lambda0_array, deltat, Nv, alpha_t0, psi_photon_psi)
    return py_derivative / py 
end

function update_D(Np::Int64, Nphoton::Int64, Δt::Real, Nv::Int64, alpha_t0::Array{Float64,2}, D_old::Real, lambda0_array::Array{Float64,1}, Qx::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, w0::Array{Float64,2},y_record::Array{Float64,2}, k_delta::Real)
    g_array = zeros(Nphoton)
    aj_mat = zeros((Nphoton, Nv))
    bj_mat = zeros((Nphoton, Nv))
    cj_mat = zeros((Nphoton, Nv))

    alpha = zeros(Nv, 1)
    alpha[:,1] = alpha_t0
    
    for τ = 1:Nphoton
        y = y_record[τ+1]
        photon_mat = get_photon_matrix_gaussian(y, xref, e_norm, interpo_xs, Np, w0, k_delta)
        psi_photon_psi = Qx' * photon_mat * Qx

        g_array[τ] = sum(get_fjD(D_old, lambda0_array, Δt, j, Nv, alpha, psi_photon_psi)^2 for j=1:Nv)
        aj_mat[τ,:] = [get_aj(j, Nv, lambda0_array, alpha, psi_photon_psi) for j=1:Nv]
        bj_mat[τ,:] = [get_bj(j, Nv, alpha, psi_photon_psi) for j=1:Nv]
        cj_mat[τ,:] = [get_cj(j, Nv, lambda0_array, alpha, psi_photon_psi) for j=1:Nv]

        expLQDT = exp.(-(D_old * lambda0_array) .* Δt)
        alpha_e_delta_t = expLQDT .* alpha
        alpha_next =  psi_photon_psi * alpha_e_delta_t
        alpha[:,1] = alpha_next / norm(alpha_next)
    end

    upper_term = sum([(1/g_array[τ]) * dot(bj_mat[τ,:], cj_mat[τ,:]) for τ = 1:Nphoton])
    bottom_term = sum([(1/g_array[τ]) * (dot(aj_mat[τ,:], bj_mat[τ,:]) - dot(cj_mat[τ,:],cj_mat[τ,:])) for τ = 1:Nphoton])
    return upper_term / (Δt * bottom_term)
end

### Analytical Get log likelihood
function get_loglikelihood_analytical(Np::Int64, Nphoton::Int64, Δt::Real, Nv::Int64, alpha_t0::Array{Float64,2}, D::Real, lambda0_array::Array{Float64,1}, Qx::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, w0::Array{Float64,2},y_record::Array{Float64,2}, k_delta::Real)
    g_array = zeros(Nphoton)

    alpha = zeros(Nv, 1)
    alpha[:,1] = alpha_t0

    for τ = 1:Nphoton
        y = y_record[τ+1]
        photon_mat = get_photon_matrix_gaussian(y, xref, e_norm, interpo_xs, Np, w0, k_delta)
        psi_photon_psi = Qx' * photon_mat * Qx

        g_array[τ] = sum(get_fjD(D, lambda0_array, Δt, j, Nv, alpha, psi_photon_psi)^2 for j=1:Nv)

        expLQDT = exp.(-(D * lambda0_array) .* Δt)
        alpha_e_delta_t = expLQDT .* alpha
        alpha_next =  psi_photon_psi * alpha_e_delta_t
        alpha[:,1] = alpha_next / norm(alpha_next)
    end
    return sum(log.(g_array))
end

function get_dldD_analytical(Np::Int64, Nphoton::Int64, Δt::Real, Nv::Int64, alpha_t0::Array{Float64,2}, D::Real, lambda0_array::Array{Float64,1}, Qx::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, w0::Array{Float64,2},y_record::Array{Float64,2}, k_delta::Real)
    g_array = zeros(Nphoton)
    dgdD_array = zeros(Nphoton)

    alpha = zeros(Nv, 1)
    alpha[:,1] = alpha_t0

    for τ = 1:Nphoton
        y = y_record[τ+1]
        photon_mat = get_photon_matrix_gaussian(y, xref, e_norm, interpo_xs, Np, w0, k_delta)
        psi_photon_psi = Qx' * photon_mat * Qx

        g_array[τ] = sum(get_fjD(D, lambda0_array, Δt, j, Nv, alpha, psi_photon_psi)^2 for j=1:Nv)
        dgdD_array[τ] = get_py_derivative(D, lambda0_array, Δt, Nv, alpha, psi_photon_psi)

        expLQDT = exp.(-(D * lambda0_array) .* Δt)
        alpha_e_delta_t = expLQDT .* alpha
        alpha_next =  psi_photon_psi * alpha_e_delta_t
        alpha[:,1] = alpha_next / norm(alpha_next)
    end
    return dot(1/g_array, dgdD_array)
end


### Numerical Get log likelihood
function get_loglikelihood_numerical(Np::Int64, Nphoton::Int64, Δt::Real, Nv::Int64, alpha_t0::Array{Float64,2}, D::Real, lambda0_array::Array{Float64,1}, Qx::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, w0::Array{Float64,2},y_record::Array{Float64,2}, k_delta::Real)
    alpha_norm_array = zeros(Nphoton)

    alpha = zeros(Nv, 1)
    alpha[:,1] = alpha_t0

    for τ = 1:Nphoton
        y = y_record[τ+1]
        photon_mat = get_photon_matrix_gaussian(y, xref, e_norm, interpo_xs, Np, w0, k_delta)
        psi_photon_psi = Qx' * photon_mat * Qx

        expLQDT = exp.(-(D * lambda0_array) .* Δt)
        alpha_e_delta_t = expLQDT .* alpha
        alpha_next =  psi_photon_psi * alpha_e_delta_t
        alpha_norm_array[τ] = norm(alpha_next)
        alpha[:,1] = alpha_next / alpha_norm_array[τ]
    end
    return 2 * sum(log.(alpha_norm_array))
end
