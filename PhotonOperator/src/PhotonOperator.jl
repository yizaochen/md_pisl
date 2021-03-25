module PhotonOperator

    using SparseArrays

    export find_nearest_point, get_photon_matrix_delta, get_photon_matrix_gaussian, get_gaussian
    
    function find_nearest_point(x, xref, e_norm, interpo_xs, Np)
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

    function get_gaussian(k_photon, xref, idx)
        mu = xref[idx]
        # Unit of k_photon: kcal/mol/angstrom^2
        sigma_photon = 1 / sqrt(2 * k_photon)
        factor1 = -1 / 2
        factor2  = 1 / ( sigma_photon * sqrt(2 * pi))
        f_x = factor2 .* exp.(factor1 .* ((xref .- mu) ./ sigma_photon).^2)
        f_x = f_x ./ sum(f_x)
        return max.(f_x, 1e-10)
    end

end
