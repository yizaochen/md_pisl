module Diffusity
using Base: Float64
using LinearAlgebra, Dierckx, PhotonOperator, Printf, Roots

include("forwardbackward_D.jl")
include("eta.jl")


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

function optimize_D(Nv::Int64, number_photon::Int64, w0::Array{Float64,2}, p_eq::Array{Float64,2}, N::Int64, Qx_prime::Array{Float64,2}, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, k_delta::Real, D_init::Real, λ_array::Array{Float64,1}, Δt::Float64)
    big_photon_mat = get_big_photon_mat(N, Nv, w0, k_delta, xref, Qx_prime)
    idx_array = [find_nearest_point(y_record[time_idx], xref, e_norm, interpo_xs, Np) for time_idx=1:number_photon]
    alpha_mat, y_beta_mat = get_alpha_beta_mat(Nv, number_photon, w0, p_eq, N, Qx_prime, D_init, λ_array, Δt, big_photon_mat, idx_array)

    function l_derivate(D::Real)
        l_derivative_mat = zeros(number_photon)
        expLQDT = exp.(-(D .* λ_array) .* Δt)
        LQDTexpLQDT = -(λ_array .* Δt) .* expLQDT
    
        # ⟨𝛼0|e{-HΔt}y1|𝛽̂1⟩
        alpha_t0 = zeros(Nv)
        alpha_t0[1] = 1
        numerator = dot(LQDTexpLQDT .* alpha_t0, y_beta_mat[:, 1])
        denominator = dot(expLQDT .* alpha_t0, y_beta_mat[:, 1])
        l_derivative_mat[1] = numerator / denominator
    
        for time_idx=2:number_photon
            numerator = dot(LQDTexpLQDT .* alpha_mat[:,time_idx-1], y_beta_mat[:, time_idx])
            denominator = dot(expLQDT .* alpha_mat[:,time_idx-1], y_beta_mat[:, time_idx])
            l_derivative_mat[time_idx] = numerator / denominator
        end
        return sum(l_derivative_mat)
    end

    return find_zero(l_derivate, D_init)
end

end # module
