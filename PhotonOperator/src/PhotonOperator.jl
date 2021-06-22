module PhotonOperator

    using Base: Int64
using SparseArrays

    export find_nearest_point, get_photon_matrix_delta, get_photon_matrix_gaussian, get_gaussian, get_p_y_given_x_mat, get_photon_matrix_gaussian_v1
    
    function find_nearest_point(x::Real, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64)
        x_left = xref[1] # The most left point
        diff = x - x_left
        n_element = floor(Int, diff / e_norm)
        node_left = x_left + n_element * e_norm
        points = node_left .+ interpo_xs
        min_idx = argmin(abs.(points .- x))
        idx = n_element * (Np - 1) + min_idx
        return idx
    end

    function find_nearest_point(x::Real, xref::Array{Float64,1}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64)
        x_left = xref[1] # The most left point
        diff = x - x_left
        n_element = floor(Int, diff / e_norm)
        node_left = x_left + n_element * e_norm
        points = node_left .+ interpo_xs
        min_idx = argmin(abs.(points .- x))
        idx = n_element * (Np - 1) + min_idx
        return idx
    end

    function get_photon_matrix_delta(x, xref, e_norm, interpo_xs, Np, w0)
        idx = find_nearest_point(x, xref, e_norm, interpo_xs, Np)
        temp_vec = zeros(size(xref))
        temp_vec[idx] = 1
        temp_vec = w0 .* temp_vec
        photon_mat = spdiagm(0 => vec(temp_vec))
        return photon_mat
    end

    function get_photon_matrix_gaussian(x, xref, e_norm, interpo_xs, Np, w0, k_photon)
        idx = find_nearest_point(x, xref, e_norm, interpo_xs, Np)
        temp_vec = get_gaussian(k_photon, xref, idx)
        temp_vec = w0 .* temp_vec
        photon_mat = spdiagm(0 => vec(temp_vec))
        return photon_mat
    end

    function get_photon_matrix_gaussian_v1(Nv::Int64, w0::Array{Float64,2}, p_y_given_x_mat::Array{Float64,2}, eigenvect_mat_prime::Array{Float64,2}, x::Real, xref::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64)
        idx = find_nearest_point(x, xref, e_norm, interpo_xs, Np)
        photon_matrix = zeros(Nv,Nv)
        for i=1:Nv
            for j=1:Nv
                psi_i = eigenvect_mat_prime[:,i]
                psi_j = eigenvect_mat_prime[:,j]
                photon_matrix[i,j] = sum(w0 .* p_y_given_x_mat[:,idx] .* psi_i .* psi_j)
            end
        end
        return photon_matrix
    end

    function get_gaussian(k_photon::Real, xref::Array{Float64,2}, idx::Int64)
        mu = xref[idx]
        # Unit of k_photon: kcal/mol/angstrom^2
        sigma_photon = 1 / sqrt(2 * k_photon)
        factor1 = -1 / 2
        factor2  = 1 / ( sigma_photon * sqrt(2 * pi))
        f_x = factor2 .* exp.(factor1 .* ((xref .- mu) ./ sigma_photon).^2)
        f_x = f_x ./ sum(f_x)
        return max.(f_x, 1e-10)
    end

    function get_gaussian(k_photon::Real, xref::Array{Float64,1}, idx::Int64)
        mu = xref[idx]
        # Unit of k_photon: kcal/mol/angstrom^2
        sigma_photon = 1 / sqrt(2 * k_photon)
        factor1 = -1 / 2
        factor2  = 1 / ( sigma_photon * sqrt(2 * pi))
        f_x = factor2 .* exp.(factor1 .* ((xref .- mu) ./ sigma_photon).^2)
        f_x = f_x ./ sum(f_x)
        return max.(f_x, 1e-10)
    end

    function get_p_y_given_x_mat(N::Int64, k_delta::Real, xref::Array{Float64,2}, w0::Array{Float64,2})
        p_y_given_x_mat = zeros(N,N)
        for row_idx=1:N
            p_y_given_x = get_gaussian(k_delta, xref, row_idx)
            p_y_given_x_mat[row_idx,:] = p_y_given_x ./ sum(w0 .* p_y_given_x)
        end
        return p_y_given_x_mat
    end

end
