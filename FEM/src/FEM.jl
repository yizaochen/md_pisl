module FEM
using LinearAlgebra, Statistics, SparseArrays

    export lglnodes, getLagrange, getLagrange_with_xref, get_fem_xref_weights_basis, fem_solve_eigen_by_pref

    """
    lglnodes(N)
    
    Computes the Legendre-Gauss-Lobatto nodes, weights and the LGL Vandermonde  matrix. The LGL nodes are the zeros of (1-x^2)*P'_N(x). Useful for numerical integration and spectral methods. 
    
    Reference on LGL nodes and weights: 
    % C. Canuto, M. Y. Hussaini, A. Quarteroni, T. A. Tang, "Spectral Methods in Fluid Dynamics," Section 2.3. Springer-Verlag 1987
    %
    % Written by Greg von Winckel - 04/17/2004
    % Contact: gregvw@chtm.unm.edu
    %
    """
    function lglnodes(N) 
        # Truncation + 1
        N1=N+1
        
        # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
        x=cos.(pi*(0:N)/N)
        
        # The Legendre Vandermonde Matrix
        # It will be used to store every term of ...
        # Legendre Polynomial
        P=zeros(N1,N1)
        
        xold = 2
        
        while maximum(abs.(x .- xold)) > eps()
            xold = x
            P[:,1] = ones(N1,1)
            P[:,2] = x
        
            for k=2:N
                factor1 = (2*k-1) * x
                term1 = factor1 .* P[:,k]
                factor2 = k - 1
                term2 = factor2 .* P[:,k-1]
                term3 = term1 .- term2
                P[:,k+1]= term3 ./ k
            end
            
            x = xold .- (x .* P[:,N1] .- P[:,N]) ./ (N1 * P[:,N1])
        end
        
        w = 2 ./ (N * N1 .* P[:,N1].^2)
        return x, w, P
    end

    """
    getLagrange(N, xratio)

    get the single element node position, weight, and Lagrange and the derivative of Lagrange
    """
    function getLagrange(N, xratio)
        N = N - 1
        x,w,P = lglnodes(N)

        x = reverse(x, dims=1)
        w = reverse(w, dims=1)
        Pend = reverse(P[:,end], dims=1)

        dL = zeros(size(P))
        
        for i=1:N+1
            term1 = Pend ./ Pend[i]
            dL[:,i] = term1 ./ (x .- x[i])
            dL[i,i] = 0
        end

        dL[1,1] = -(N*(N+1)) / 4
        dL[end,end] = N*(N+1) / 4
        L = Diagonal(x)

        # Scale by xratio
        x = x .* xratio
        w = w .* xratio
        dL = dL ./ xratio
        return x, w, dL, L
    end

    """
    getLagrange_with_xref(N, xratio, xref)

    get the single element node position, weight, and Lagrange and the derivative of Lagrange
    """
    function getLagrange_with_xref(N, xratio, xref)
        N = N - 1
        x,w,P = lglnodes(N)
        
        x = reverse(x, dims=1)
        w = reverse(w, dims=1)
        Pend = reverse(P[:,end], dims=1)
        
        xref = xref .- mean(xref)
        xref = xref ./ xratio
        
        L = ones(length(xref), N+1)
        dL = zeros(length(xref), N+1)
        
        for i=1:N+1
            for j=1:N+1
                if i != j
                    term1 = (xref .- x[j]) ./ (x[i] - x[j])
                    L[:,i] = L[:,i] .* term1
                end
            end
        end
        
        for i=1:N+1
            for k =1:N+1
                if k != i
                    temp = ones(size(xref))
                    for j=1:N+1
                        if (i != j) && (j != k)
                            term1 = (xref .- x[j]) ./ (x[i] - x[j])
                            temp = temp .* term1
                        end
                        
                        if (i != j) && (j == k)
                            term1 = x[i] - x[j]
                            temp = temp ./ term1
                        end
                    end
                    dL[:,i] = dL[:,i] + temp
                end
            end
        end
        # Scale by xratio
        x = x .* xratio
        w = w .* xratio
        dL = dL ./ xratio
        return x, w, dL, L
    end


    """
    get_fem_xref_weights_basis(Nh, Np)
    
    Create Finite Element nodes, weights, and basis function
    Nh: The number of Spectral element
    Np: The order of polynomial which used to interpolate and integration
    xratio: The scale factor for affine transformation. 
            Original domain: [-1, 1].
    xavg: The center of x-domain. 
          Now the whole domain is [xavg-xratio, xavg+xratio].
    """
    function get_fem_xref_weights_basis(Nh, Np, xratio, xavg)
        N  = Nh*Np - Nh + 1 # Total number of nodes

        # x: GLL collocation points, w: integration weights
        # Ldx: The derivative of Lagrange polynomial, L: Lagrange polynomial
        x, w, Ldx, L = getLagrange(Np,xratio/Nh) 

        xref = zeros(N,1)    # grid points of whole domain
        w0   = zeros(N,1)    # integration kernel (int=sum(w0.*f(x)))
        #Tndx = spzeros(N, N) # Create a sparse zero array, which will store the derivative of basis functions
        
        for i=0:Nh-1
            idx_array = 1+i*(Np-1):i*(Np-1)+Np
            
            # Set x positions 
            term1 = (2 * xratio/Nh * i) + (xratio/Nh)
            xref[idx_array] = x .+ term1
            
            # Set weight for integration
            w0[idx_array] = w0[idx_array] .+ w
        
            # Set the derivative of basis functions
            #Tndx[idx_array, idx_array] = Tndx[idx_array, idx_array] .+ Ldx
        end
        
        #=
        for i=0:Nh-1
            idx = Np + i * (Np-1)
            Tndx[idx,:] = Tndx[idx,:] ./ 2
        end
        Tndx[end,:] = Tndx[end,:] .* 2
        =#

        # Rescale from 0 to 1 --> 0.5 to 1.5
        term1 = xavg - xratio
        xref = xref .+ term1
        return N, xref, w0, Ldx, w
    end

    """
    get_pref(Vref, w0)

    Get the equilibrium probability according to a potential function Vref
    """
    function get_pref(Vref, w0)
        pref = exp.(-Vref)
        pref = max.(pref,1e-10)
        sum_factor = sum(w0 .* pref)
        pref = pref ./ sum_factor
        return pref
    end


    function get_rhoref(Vref, w0)
        pref = exp.(-Vref)
        pref = max.(pref,1e-10)
        sum_factor = sum(w0 .* pref)
        pref = pref ./ sum_factor
        rho_ref = sqrt.(pref) 
        return rho_ref
    end

    """
    fem_solve_eigen_by_pref(Nh, Np, xratio, xavg, pref, D, Nv)
    
    Using spectral finite element method to diagonalize the H operator
    """
    function fem_solve_eigen_by_pref(Nh, Np, xratio, xavg, pref, D, Nv)
        N, xref, w0, Ldx, w = get_fem_xref_weights_basis(Nh, Np, xratio, xavg)
        D_array = D .* ones(size(xref))

        ## Hamiltonian Matrix K
        Ke = spzeros(N, N) # Sparse Array allocate
         
        for k=0:Nh-1
            idx_array = 1+k*(Np-1):k*(Np-1)+Np
            temp_diag = Diagonal(D_array[idx_array] .* w .* pref[idx_array])
            temp_mat = Ldx' * temp_diag * Ldx
            Ke[idx_array, idx_array] = Ke[idx_array, idx_array] .+ temp_mat
        end
        
        # Overlap Matrix S
        temp = w0 .* pref
        Se = spdiagm(0 => vec(temp))

        # Sparse to dense
        Ke_dense = Array(Ke)
        Se_dense = Array(Se)
        
        # Find Eigenvalue and eigenvector
        F = eigen(Ke_dense, Se_dense)
        LQ = F.values
        Q = F.vectors
        
        # Process Eigenvalues and Eigenvectors
        LQ = real.(LQ) # Get real part of eigenvalues
        iLQ = sortperm(LQ) # iLQ: index of sorted eigenvalues
        LQ = LQ[iLQ]
        LQ[1] = 0
        Q = Q[:, iLQ] # Sorted eigenvectors by index
        Q = real.(Q)  # Eq.(36)

        LQ = LQ[1:Nv]
        
        # We want the true eigenvector of Eq.(33), which we name as Qx
        Qx = Q[:, 1:Nv]
        rho_eq = pref .^ (1/2) # p_eq^(1/2)
        rho_eq_diag_mat = spdiagm(0 => vec(rho_eq))
        Qx = rho_eq_diag_mat * Qx # Psi, Eq. (35)

        # Normalize
        for i=2:Nv
            Qx[:,i] = Qx[:,i] ./ sqrt(sum(w0 .* Qx[:,i] .* Qx[:,i]))
        end
        return LQ, Qx, rho_eq
    end

    function fem_solve_eigen_by_pref_old(Nh, Np, xratio, xavg, pref, D, Nv)
        N, xref, w0, Ldx, w = get_fem_xref_weights_basis(Nh, Np, xratio, xavg)
        D_array = D .* ones(size(xref))

        ## Hamiltonian Matrix K
        Ke = spzeros(N, N) # Sparse Array allocate
         
        for k=0:Nh-1
            idx_array = 1+k*(Np-1):k*(Np-1)+Np
            temp_diag = Diagonal(D_array[idx_array] .* w .* pref[idx_array])
            temp_mat = Ldx' * temp_diag * Ldx
            Ke[idx_array, idx_array] = Ke[idx_array, idx_array] .+ temp_mat
        end
        
        # Overlap Matrix S
        temp = w0 .* pref
        Se = spdiagm(0 => vec(temp))
        
        # Find Eigenvalue and eigenvector
        vstart=ones(size(Se,1))
        LQ, Q = eigs(Ke, Se, nev=Nv, which=:SM, maxiter=300, v0=vstart)
        
        # Process Eigenvalues and Eigenvectors
        LQ = real.(LQ) # Get real part of eigenvalues
        iLQ = sortperm(LQ) # iLQ: index of sorted eigenvalues
        LQ = LQ[iLQ]
        LQ[1] = 0
        Q = Q[:, iLQ] # Sorted eigenvectors by index
        Q = real.(Q)  # Eq.(36)
        
        # We want the true eigenvector of Eq.(33), which we name as Qx
        Qx = Q[:, 1:Nv]
        rho_eq = pref .^ (1/2) # p_eq^(1/2)
        rho_eq_diag_mat = spdiagm(0 => vec(rho_eq))
        Qx = rho_eq_diag_mat * Qx # Psi, Eq. (35)
        return LQ, Qx, rho_eq
    end
    
    function get_coefficients_by_proj(w0, alpha, Qx, Nv)
        c_array = ones(Nv)
        temp = w0 .* alpha
        for idx_eigv in 1:Nv
            c_array[idx_eigv] = sum(temp .* Qx[:, idx_eigv])
        end
        return c_array
    end


    function rebuild_p0_by_Qx_carray(Nv, c_array, Qx, rho_eq)
        N = size(Qx)[1]
        temp = zeros(N)
        for idx_eigv in 1:Nv
            temp = temp .+ (c_array[idx_eigv] .* Qx[:, idx_eigv])
        end
        return temp .* rho_eq
    end

    function get_residual(f, f_appr)
        return sum((f .- f_appr).^2)
    end

end # module
