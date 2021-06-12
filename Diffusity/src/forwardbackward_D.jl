
function initialize_mat_vec(Nv::Int64, number_photon::Int64)
    alpha_mat = zeros(Nv,number_photon)
    beta_mat = zeros(Nv,number_photon)
    Anorm_vec = ones(1,number_photon) # sqrt(c_array)
    r_sqrt_array = ones(1, number_photon)
    return alpha_mat, beta_mat, Anorm_vec, r_sqrt_array
end

function get_rho_s1(w0::Array{Float64,2}, p_eq::Array{Float64,2}, weight_Qx::Array{Float64,2})
    p_x1 = p_eq / sum(w0 .* p_eq)
    return transpose(weight_Qx) * (sqrt.(p_x1))
end

function get_weight_Qx(N::Int64, Nv::Int64, w0::Array{Float64,2}, Qx::Array{Float64,2})
    weight_Qx = zeros(N, Nv)
    for i = 1:Nv
        weight_Qx[:, i] = w0 .* Qx[:, i]
    end
    return weight_Qx
end

function proj_vector_from_eigenspace_to_xspace(Qx::Array{Float64,2}, vector::Array{Float64, 2})
    return Qx * vector
end

function proj_vector_from_eigenspace_to_xspace(Qx::Array{Float64,2}, vector::Array{Float64, 1})
    return Qx * vector
end

function forward_backward_v0(Nv::Int64, number_photon::Int64, w0::Array{Float64,2}, p_eq::Array{Float64,2}, N::Int64, Qx_prime::Array{Float64,2}, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, k_delta::Real, D_guess::Real, eigenvalues_prime::Array{Float64,1}, save_freq::Float64)
    alpha_mat, beta_mat, Anorm_vec, r_sqrt_array = initialize_mat_vec(Nv, number_photon)
    weight_Qx = get_weight_Qx(N, Nv, w0, Qx_prime)
    rho_s1 = get_rho_s1(w0::Array{Float64,2}, p_eq::Array{Float64,2}, weight_Qx::Array{Float64,2})
    beta_N_bra = transpose(weight_Qx) * ones(N,1)
    expLQDT = exp.(-(D_guess * eigenvalues_prime) .* save_freq)

    alpha_mat, Anorm_vec = forward_v0(Nv, number_photon, rho_s1, y_record, xref, e_norm, interpo_xs, Np, w0, k_delta, Qx_prime, alpha_mat, Anorm_vec, expLQDT)
    beta_mat, r_sqrt_array = backward_v0(Nv, number_photon, beta_N_bra, y_record, xref, e_norm, interpo_xs, Np, w0, k_delta, Qx_prime, beta_mat, alpha_mat, r_sqrt_array, expLQDT)
    return alpha_mat, beta_mat, Anorm_vec, r_sqrt_array
end

function forward_v0(Nv::Int64, number_photon::Int64, rho_s1::Array{Float64,2}, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, w0::Array{Float64,2}, k_delta::Real, Qx_prime::Array{Float64,2}, alpha_mat::Array{Float64,2}, Anorm_vec::Array{Float64,2}, expLQDT::Array{Float64,1})
    alpha_hat_prev = zeros((Nv,1))
    alpha_hat_prev[:,1] = rho_s1

    for time_idx = 1:number_photon   
        photon_mat = get_photon_matrix_gaussian(y_record[time_idx], xref, e_norm, interpo_xs, Np, w0, k_delta)
        psi_photon_psi = Qx_prime' * photon_mat * Qx_prime
    
        alpha_bra = psi_photon_psi * alpha_hat_prev    
        Anorm_vec[time_idx] = norm(alpha_bra)
        
        # Normalization
        alpha_hat_bra = alpha_bra ./ Anorm_vec[time_idx]
        alpha_mat[:,time_idx] = alpha_hat_bra
        
        # Time propagation
        alpha_hat_prev = expLQDT .* alpha_hat_bra
        alpha_hat_prev[1] = sign(alpha_hat_bra[1]) * sqrt(1 - sum((alpha_hat_prev[2:end]).^2))
    end  
    return alpha_mat, Anorm_vec
end

function backward_v0(Nv::Int64, number_photon::Int64, beta_N_bra::Array{Float64,2}, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, w0::Array{Float64,2}, k_delta::Real, Qx_prime::Array{Float64,2}, beta_mat::Array{Float64,2}, alpha_mat::Array{Float64,2}, r_sqrt_array::Array{Float64,2}, expLQDT::Array{Float64,1})
    beta_hat_next = zeros((Nv,1))
    beta_hat_next[:,1] = beta_N_bra

    for time_idx = number_photon:-1:1
        photon_mat = get_photon_matrix_gaussian(y_record[time_idx], xref, e_norm, interpo_xs, Np, w0, k_delta)
        psi_photon_psi = Qx_prime' * photon_mat * Qx_prime

        # Photon operation
        y_beta_hat_next = psi_photon_psi * beta_hat_next

        # Time propagation
        edt_y_beta_hat_next = expLQDT .* y_beta_hat_next
        edt_y_beta_hat_next[1] = sign(edt_y_beta_hat_next[1]) * sqrt(dot(y_beta_hat_next, y_beta_hat_next) - sum((edt_y_beta_hat_next[2:end]).^2))

        # Normalization
        alpha_hat_square = proj_vector_from_eigenspace_to_xspace(Qx_prime, alpha_mat[:,time_idx]) .^ 2
        x_edt_y_beta_hat_next_square = proj_vector_from_eigenspace_to_xspace(Qx_prime, edt_y_beta_hat_next) .^ 2
        r_sqrt_array[time_idx] = sqrt(sum(w0 .* alpha_hat_square .* x_edt_y_beta_hat_next_square))

        beta_hat_next = edt_y_beta_hat_next ./ r_sqrt_array[time_idx]
        beta_mat[:,time_idx] = beta_hat_next
    end
    return beta_mat, r_sqrt_array
end

function get_gamma(N::Int64, Nv::Int64, w0::Array{Float64,2}, alpha_mat::Array{Float64,2}, beta_mat::Array{Float64,2}, time_idx::Int64, Qx_prime::Array{Float64,2})
    weight_Qx = get_weight_Qx(N, Nv, w0, Qx_prime)
    alpha_hat_x_square = proj_vector_from_eigenspace_to_xspace(Qx_prime, alpha_mat[:,time_idx]) .^ 2
    beta_hat_x_square = proj_vector_from_eigenspace_to_xspace(Qx_prime, beta_mat[:,time_idx]) .^ 2
    gamma_x = alpha_hat_x_square .* beta_hat_x_square
    gamma_bra = transpose(weight_Qx) * sqrt.(gamma_x)
    gamma_s = gamma_bra .^ 2
    return gamma_x, gamma_s
end

function get_update_A()
    println("Test")
end