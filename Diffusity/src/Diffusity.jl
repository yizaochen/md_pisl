module Diffusity
using LinearAlgebra, Dierckx

function sigmoid(x_center::Real, x::Real, scale_factor::Real, translation_factor::Real)
    z = x - x_center
    return (one(z) / (one(z) + exp(-z))) * scale_factor + translation_factor
end

function linear(xleft::Real, xright::Real, yleft::Real, yright::Real, x::Real, c::Real)
    """
    model: y = mx + c
    """
    m = (yright - yleft) / (xright - xleft)
    return m * x + c
end

function get_eigv_derivative(xref::Array{Float64,2}, idx_eigv::Int64, Qx::Array{Float64,2}, N::Int64)
    eigv_spl = Spline1D(xref[:,1], Qx[:, idx_eigv])
    eigv_derivative = zeros(N)
    for idx = 1:N
        eigv_derivative[idx] = derivative(eigv_spl, xref[idx])
    end
    return eigv_derivative
end

function get_F_derivative(xref::Array{Float64,2}, F_array::Array{Float64,2}, N::Int64)
    F_spl = Spline1D(xref[:,1], F_array[:, 1])
    F_derivative = zeros(N)
    for idx = 1:N
        F_derivative[idx] = derivative(F_spl, xref[idx])
    end
    return F_derivative
end

function get_eta_derivative(xref::Array{Float64,2}, eta_array::Array{Float64,2}, N::Int64)
    eta_spl = Spline1D(xref[:,1], eta_array[:, 1])
    eta_derivative = zeros(N)
    for idx = 1:N
        eta_derivative[idx] = derivative(eta_spl, xref[idx])
    end
    return eta_derivative
end

function solven_eigen_H(Nv::Int64, xref::Array{Float64,2}, Qx_H0::Array{Float64,2}, eigenvalue_H0::Array{Float64,1}, N::Int64, F_xref::Array{Float64,2}, eta_array::Array{Float64,2})
    # Get required function
    F_derivative = get_F_derivative(xref, F_xref, N)
    eta_derivative = get_eta_derivative(xref, eta_array, N)

    K_matrix = zeros(Nv, Nv)
    for i_idx = 1:Nv
        eigv_i_derivative = get_eigv_derivative(xref, i_idx, Qx_H0, N)
        for j_idx = 1:Nv
            eigv_j_derivative = get_eigv_derivative(xref, j_idx, Qx_H0, N)
            
            first_term = sum(eta_array .* eigv_i_derivative .* eigv_j_derivative)        
            second_term = sum((F_derivative .* eta_array) .* Qx_H0[:, i_idx] .* Qx_H0[:, j_idx]) / 2
            third_term = sum(eta_array .* F_xref .* F_xref .* Qx_H0[:, i_idx] .* Qx_H0[:, j_idx]) / 4
            fourth_term = sum(eta_derivative .* F_xref .* Qx_H0[:, i_idx] .* Qx_H0[:, j_idx]) / 2
            final_result = first_term + second_term + third_term + fourth_term
            
            if i_idx == j_idx
                K_matrix[i_idx, j_idx] = final_result + eigenvalue_H0[i_idx]
            else
                K_matrix[i_idx, j_idx] = final_result
                K_matrix[j_idx, i_idx] = final_result
            end
        end
    end

    F = eigen(K_matrix)
    eigenvalues_H = F.values
    K_eigenvect_mat = F.vectors

    Qx_H = Qx_H0 * transpose(K_eigenvect_mat)
    eigenvalues_H[1] = 0.
    return eigenvalues_H, Qx_H
end

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


end # module
