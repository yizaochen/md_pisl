
function initialize_mat_vec(Nv::Int64, number_photon::Int64)
    alpha_mat = zeros(Nv,number_photon)
    beta_mat = zeros(Nv,number_photon)
    Anorm_vec = ones(1,number_photon) # sqrt(c_array)
    r_sqrt_array = ones(1, number_photon)
    return alpha_mat, beta_mat, Anorm_vec, r_sqrt_array
end

function initialize_mat_vec_v1(Nv::Int64, number_photon::Int64)
    alpha_mat = zeros(Nv,number_photon)
    beta_mat = zeros(Nv,number_photon)
    Anorm_vec = ones(1,number_photon) # sqrt(c_array)
    return alpha_mat, beta_mat, Anorm_vec
end

function initialize_mat_vec_onlyalpha_v1(Nv::Int64, number_photon::Int64)
    alpha_mat = zeros(Nv,number_photon)
    Anorm_vec = ones(1,number_photon) # sqrt(c_array)
    return alpha_mat, Anorm_vec
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

function forward_backward_v0(Nv::Int64, number_photon::Int64, w0::Array{Float64,2}, p_eq::Array{Float64,2}, N::Int64, Qx_prime::Array{Float64,2}, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, k_delta::Real, D_guess::Array{Float64,1}, eigenvalues_prime::Array{Float64,1}, save_freq::Float64)
    alpha_mat, beta_mat, Anorm_vec, r_sqrt_array = initialize_mat_vec(Nv, number_photon)
    weight_Qx = get_weight_Qx(N, Nv, w0, Qx_prime)
    rho_s1 = get_rho_s1(w0, p_eq, weight_Qx)
    beta_N_bra = transpose(weight_Qx) * ones(N,1)
    expLQDT = exp.(-(D_guess .* eigenvalues_prime) .* save_freq)

    alpha_mat, Anorm_vec = forward_v0(Nv, number_photon, rho_s1, y_record, xref, e_norm, interpo_xs, Np, w0, k_delta, Qx_prime, alpha_mat, Anorm_vec, expLQDT)
    beta_mat, r_sqrt_array = backward_v0(Nv, number_photon, beta_N_bra, y_record, xref, e_norm, interpo_xs, Np, w0, k_delta, Qx_prime, beta_mat, alpha_mat, r_sqrt_array, expLQDT)
    return alpha_mat, beta_mat, Anorm_vec, r_sqrt_array
end

function forward_backward_v1(Nv::Int64, number_photon::Int64, w0::Array{Float64,2}, p_eq::Array{Float64,2}, N::Int64, Qx_prime::Array{Float64,2}, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, k_delta::Real, D_guess::Array{Float64,1}, eigenvalues_prime::Array{Float64,1}, save_freq::Float64)
    alpha_mat, beta_mat, Anorm_vec = initialize_mat_vec_v1(Nv, number_photon)
    weight_Qx = get_weight_Qx(N, Nv, w0, Qx_prime)
    rho_s1 = get_rho_s1(w0, p_eq, weight_Qx)
    expLQDT = exp.(-(D_guess .* eigenvalues_prime) .* save_freq)
    big_photon_mat = get_big_photon_mat(N, Nv, w0, k_delta, xref, Qx_prime)
    idx_array = [find_nearest_point(y_record[time_idx], xref, e_norm, interpo_xs, Np) for time_idx=1:number_photon]

    alpha_mat, Anorm_vec = forward_v1(Nv, number_photon, rho_s1, alpha_mat, Anorm_vec, expLQDT, big_photon_mat, idx_array)
    beta_mat = backward_v1(Nv, number_photon, beta_mat, Anorm_vec, expLQDT, big_photon_mat, idx_array)
    return alpha_mat, beta_mat, Anorm_vec
end

function scan_l_by_vary_D(Nv::Int64, number_photon::Int64, w0::Array{Float64,2}, p_eq::Array{Float64,2}, N::Int64, Qx_prime::Array{Float64,2}, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, k_delta::Real, D_array::Array{Float64,1}, eigenvalues_prime::Array{Float64,1}, save_freq::Float64)
    weight_Qx = get_weight_Qx(N, Nv, w0, Qx_prime)
    rho_s1 = get_rho_s1(w0, p_eq, weight_Qx)
    big_photon_mat = get_big_photon_mat(N, Nv, w0, k_delta, xref, Qx_prime)
    idx_array = [find_nearest_point(y_record[time_idx], xref, e_norm, interpo_xs, Np) for time_idx=1:number_photon]
    l_array = zeros(length(D_array))
    idx = 1
    for D_value in D_array
        D_guess = D_value * ones(Nv)
        expLQDT = exp.(-(D_guess .* eigenvalues_prime) .* save_freq)
        alpha_mat, Anorm_vec = initialize_mat_vec_onlyalpha_v1(Nv, number_photon)
        alpha_mat, Anorm_vec = forward_v1(Nv, number_photon, rho_s1, alpha_mat, Anorm_vec, expLQDT, big_photon_mat, idx_array)
        l_array[idx] = sum(log.(Anorm_vec))
        idx += 1
    end
    return l_array
end

function calculate_Q(Nv::Int64, number_photon::Int64, w0::Array{Float64,2}, p_eq::Array{Float64,2}, N::Int64, Qx_prime::Array{Float64,2}, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, k_delta::Real, D_value::Real, eigenvalues_prime::Array{Float64,1}, save_freq::Float64)
    D_guess = D_value * ones(Nv)
    alpha_mat, beta_mat, Anorm_vec = initialize_mat_vec_v1(Nv, number_photon)
    weight_Qx = get_weight_Qx(N, Nv, w0, Qx_prime)
    rho_s1 = get_rho_s1(w0, p_eq, weight_Qx)
    expLQDT = exp.(-(D_guess .* eigenvalues_prime) .* save_freq)
    big_photon_mat = get_big_photon_mat(N, Nv, w0, k_delta, xref, Qx_prime)
    idx_array = [find_nearest_point(y_record[time_idx], xref, e_norm, interpo_xs, Np) for time_idx=1:number_photon]

    alpha_mat, Anorm_vec = forward_v1(Nv, number_photon, rho_s1, alpha_mat, Anorm_vec, expLQDT, big_photon_mat, idx_array)
    beta_mat = backward_v1(Nv, number_photon, beta_mat, Anorm_vec, expLQDT, big_photon_mat, idx_array)
    return alpha_mat, beta_mat
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

function forward_v1(Nv::Int64, number_photon::Int64, rho_s1::Array{Float64,2}, alpha_mat::Array{Float64,2}, Anorm_vec::Array{Float64,2}, expLQDT::Array{Float64,1}, big_photon_mat::Array{Float64,3}, idx_array::Array{Int64,1})
    alpha_hat_prev = zeros(1,Nv)
    alpha_hat_prev[1,:] = rho_s1
    expLQDT = transpose(expLQDT)

    for time_idx = 1:number_photon 
        psi_photon_psi = big_photon_mat[:,:,idx_array[time_idx]]
    
        alpha_bra = alpha_hat_prev * psi_photon_psi    
        Anorm_vec[time_idx] = alpha_bra[1]
        
        # Normalization
        alpha_hat_bra = alpha_bra ./ Anorm_vec[time_idx]
        alpha_mat[:,time_idx] = alpha_hat_bra
        
        # Time propagation
        alpha_hat_prev = alpha_hat_bra .* expLQDT
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

function backward_v1(Nv::Int64, number_photon::Int64, beta_mat::Array{Float64,2}, Anorm_vec::Array{Float64,2}, expLQDT::Array{Float64,1}, big_photon_mat::Array{Float64,3}, idx_array::Array{Int64,1})
    beta_hat_next = zeros(Nv,1)
    beta_hat_next[1,1] = 1

    for time_idx = number_photon:-1:1
        beta_mat[:,time_idx] = beta_hat_next[:,1]
        psi_photon_psi = big_photon_mat[:,:,idx_array[time_idx]]

        # Photon operation
        y_beta_hat_next = psi_photon_psi * beta_hat_next

        # Time propagation
        edt_y_beta_hat_next = expLQDT .* y_beta_hat_next

        # Normalization
        beta_hat_next = edt_y_beta_hat_next ./ Anorm_vec[time_idx]
    end
    return beta_mat
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

function get_p_s_prev_s_i_given_y1_to_yi(D_guess::Array{Float64,1}, eigenvalues_prime::Array{Float64,1}, save_freq::Float64, alpha_mat::Array{Float64,2}, time_idx::Int64, Nv::Int64)
    # <α_{n-1}|   and <α_{n-1}|e^{-HΔt}
    expLQDT = exp.(-(D_guess .* eigenvalues_prime) .* save_freq)
    alpha_hat_prev = alpha_mat[:,time_idx-1]
    alpha_hat_prev_edt = expLQDT .* alpha_hat_prev
    alpha_hat_prev_edt[1] = sign(alpha_hat_prev_edt[1]) * sqrt(1 - sum((alpha_hat_prev_edt[2:end]).^2))

    col_vec = zeros(Nv,1)
    row_vec = zeros(1,Nv)
    col_vec[:,1] = alpha_hat_prev .^ 2
    row_vec[1,:] = alpha_hat_prev_edt .^ 2
    return col_vec * row_vec
end

function get_p_x_prev_x_i_given_y1_to_yi(D_guess::Array{Float64,1}, eigenvalues_prime::Array{Float64,1}, save_freq::Float64, alpha_mat::Array{Float64,2}, time_idx::Int64, N::Int64, Qx_prime::Array{Float64,2})
    # <α_{n-1}|   and <α_{n-1}|e^{-HΔt}
    expLQDT = exp.(-(D_guess .* eigenvalues_prime) .* save_freq)
    alpha_hat_prev = alpha_mat[:,time_idx-1]
    alpha_hat_prev_edt = expLQDT .* alpha_hat_prev
    alpha_hat_prev_edt[1] = sign(alpha_hat_prev_edt[1]) * sqrt(1 - sum((alpha_hat_prev_edt[2:end]).^2))

    col_vec = zeros(N,1)
    row_vec = zeros(1,N)
    col_vec[:,1] = proj_vector_from_eigenspace_to_xspace(Qx_prime, alpha_hat_prev) .^ 2
    row_vec[1,:] = proj_vector_from_eigenspace_to_xspace(Qx_prime, alpha_hat_prev_edt) .^ 2
    return col_vec * row_vec
end

function get_yi_beta_i(time_idx::Int64, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, w0::Array{Float64,2}, k_delta::Real, Qx_prime::Array{Float64,2}, beta_mat::Array{Float64,2})
    # y_n|β_n> 
    photon_mat = get_photon_matrix_gaussian(y_record[time_idx], xref, e_norm, interpo_xs, Np, w0, k_delta)
    psi_photon_psi = Qx_prime' * photon_mat * Qx_prime

    beta_hat = beta_mat[:,time_idx]
    y_beta_hat = psi_photon_psi * beta_hat
    return y_beta_hat ./ norm(y_beta_hat) # Back to normal prob scale
end

function get_xi_s_space(D_guess::Array{Float64,1}, eigenvalues_prime::Array{Float64,1}, save_freq::Float64, alpha_mat::Array{Float64,2}, time_idx::Int64, Nv::Int64, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, w0::Array{Float64,2}, k_delta::Real, Qx_prime::Array{Float64,2}, beta_mat::Array{Float64,2})
    p_s_prev_s_i_mat = get_p_s_prev_s_i_given_y1_to_yi(D_guess, eigenvalues_prime, save_freq, alpha_mat, time_idx, Nv)
    y_beta_hat_after_norm = get_yi_beta_i(time_idx, y_record, xref, e_norm, interpo_xs, Np, w0, k_delta, Qx_prime, beta_mat)
    y_beta_hat_after_norm_square = y_beta_hat_after_norm .^ 2

    # First calculate xi-matrix 
    p_s_prev_si_given_Y = zeros(Nv,Nv)
    for row_idx = 1:Nv
        p_s_prev_si_given_Y[row_idx,:] = p_s_prev_s_i_mat[row_idx, :] .* y_beta_hat_after_norm_square
    end
    
    # Start to calculate the normalization constant for matrix
    p_s_prev_marginal_vec_given_Y = zeros(Nv,1)
    for idx = 1:Nv
        p_s_prev_marginal_vec_given_Y[idx] = sum(p_s_prev_si_given_Y[idx,:]) # \int_x5 p(x4,x5|Y) dx5 = p(x4|Y)
    end
    return p_s_prev_si_given_Y ./ sum(p_s_prev_marginal_vec_given_Y) # Normalize the matrix
end

function get_xi_x_space(D_guess::Array{Float64,1}, eigenvalues_prime::Array{Float64,1}, save_freq::Float64, alpha_mat::Array{Float64,2}, time_idx::Int64, N::Int64, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, w0::Array{Float64,2}, k_delta::Real, Qx_prime::Array{Float64,2}, beta_mat::Array{Float64,2})
    p_x_prev_x_i_mat = get_p_x_prev_x_i_given_y1_to_yi(D_guess, eigenvalues_prime, save_freq, alpha_mat, time_idx, N, Qx_prime)
    y_beta_hat_after_norm = get_yi_beta_i(time_idx, y_record, xref, e_norm, interpo_xs, Np, w0, k_delta, Qx_prime, beta_mat)
    y_beta_hat_after_norm_x_square = proj_vector_from_eigenspace_to_xspace(Qx_prime, y_beta_hat_after_norm) .^ 2
    
    # First calculate xi-matrix 
    p_x_prev_xi_given_Y = zeros(N,N)
    for row_idx = 1:N
        p_x_prev_xi_given_Y[row_idx,:] = p_x_prev_x_i_mat[row_idx, :] .* y_beta_hat_after_norm_x_square
    end
    
    # Start to calculate the normalization constant for matrix
    p_x_prev_marginal_vec_given_Y = zeros(N,1)
    for idx = 1:N
        p_x_prev_marginal_vec_given_Y[idx] = sum(w0 .* p_x_prev_xi_given_Y[idx,:]) # \int_x5 p(x4,x5|Y) dx5 = p(x4|Y)
    end
    return p_x_prev_xi_given_Y ./ sum(w0 .* p_x_prev_marginal_vec_given_Y) # Normalize the matrix
end

function get_xi_store_mat(Nv::Int64, number_photon::Int64, D_guess::Array{Float64,1}, eigenvalues_prime::Array{Float64,1}, save_freq::Float64, alpha_mat::Array{Float64,2}, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, w0::Array{Float64,2}, k_delta::Real, Qx_prime::Array{Float64,2}, beta_mat::Array{Float64,2})
    xi_store_mat = zeros(Nv,Nv,number_photon)
    for photon_idx=2:number_photon
        xi_store_mat[:,:,photon_idx] = get_xi_s_space(D_guess, eigenvalues_prime, save_freq, alpha_mat, photon_idx, Nv, y_record, xref, e_norm, interpo_xs, Np, w0, k_delta, Qx_prime, beta_mat)
    end
    return xi_store_mat
end

function get_A_mat(Nv::Int64, xi_store_mat::Array{Float64,3}, number_photon::Int64)
    A_mat = zeros(Nv,Nv)
    for j=1:Nv
        for k=1:Nv
            numerator = sum([xi_store_mat[j,k,photon_idx] for photon_idx=2:number_photon])
            denominator = 0.
            for l=1:Nv
                denominator += sum([xi_store_mat[j,l,photon_idx] for photon_idx=2:number_photon])
            end
            A_mat[j,k] = numerator / denominator
        end
    end
    return transpose(A_mat)
end

function get_A_first_column(Nv::Int64, xi_store_mat::Array{Float64,3}, number_photon::Int64)
    k = 1
    A_first_row = zeros(Nv)
    for j=1:Nv
        numerator = sum([xi_store_mat[j,k,photon_idx] for photon_idx=2:number_photon])
        denominator = 0.
        for l=1:Nv
            denominator += sum([xi_store_mat[j,l,photon_idx] for photon_idx=2:number_photon])
        end
        A_first_row[j] = numerator / denominator
    end
    return A_first_row
end

function get_D_by_A_first_row(Nv::Int64, A_first_row::Array{Float64,1}, eigenvalues_prime::Array{Float64,1}, save_freq::Float64)
    D_array = ones(Nv)
    for eigv_idx=2:Nv
        D_array[eigv_idx] = log(1-A_first_row[eigv_idx]) / (-2 * eigenvalues_prime[eigv_idx] * save_freq)
    end
    return D_array
end

function EM_D_by_transition_matrix(n_iter::Int64, Nv::Int64, number_photon::Int64, w0::Array{Float64,2}, p_eq::Array{Float64,2}, N::Int64, Qx_prime::Array{Float64,2}, y_record::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, k_delta::Real, D_guess::Array{Float64,1}, eigenvalues_prime::Array{Float64,1}, save_freq::Float64)
    D_em_mat = zeros(n_iter, Nv)
    log_l_mat = zeros(n_iter)
    for iter_id=1:n_iter
        println(@sprintf "Start EM-Step: %d" iter_id)
        alpha_mat, beta_mat, Anorm_vec, r_sqrt_array = forward_backward_v0(Nv, number_photon, w0, p_eq, N, Qx_prime, y_record, xref, e_norm, interpo_xs, Np, k_delta, D_guess, eigenvalues_prime, save_freq)

        xi_store_mat = get_xi_store_mat(Nv, number_photon, D_guess, eigenvalues_prime, save_freq, alpha_mat, y_record, xref, e_norm, interpo_xs, Np, w0, k_delta, Qx_prime, beta_mat)

        A_first_row = get_A_first_column(Nv, xi_store_mat, number_photon)
        D_em_array = get_D_by_A_first_row(Nv, A_first_row, eigenvalues_prime, save_freq)

        D_em_mat[iter_id,:] = D_em_array
        D_guess[:] = D_em_array

        log_l_mat[iter_id] = sum(log.(Anorm_vec))
    end
    return D_em_mat, log_l_mat
end
