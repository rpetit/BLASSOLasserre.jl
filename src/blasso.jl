export blasso, solve!, example

mutable struct BLASSO{T<:Real, PT<:AbstractPolynomialLike}
    y::Vector{T}  # observations
    φ::Vector{PT}  # measurement functions
    λ::Real  # regularization parameter
    domain::AbstractSemialgebraicSet

    μ::Union{AbstractMeasureLike, Nothing}
    objvalue::Union{Float64, Nothing}
    dualsol::Union{Vector{Float64}, Nothing}
    mommat::Union{MomentMatrix, Nothing}
end

function blasso(y::Vector{T}, φ::Vector{PT},
                λ::Real, domain::AbstractSemialgebraicSet) where {T<:Real,
                                                                  PT<:AbstractPolynomialLike}
    BLASSO(y, φ, λ, domain, nothing, nothing, nothing, nothing)
end

function solve!(blasso::BLASSO, ϵ::Real, maxdeg::Union{String, Int64}="defaut")
    model = SOSModel(with_optimizer(Mosek.Optimizer, QUIET=true))

    y = blasso.y
    φ = blasso.φ
    m = length(y)

    @variable(model, p[1:m])  # dual variable

    if maxdeg == "defaut"
        maxdeg = max([maxdegree(monomials(φ[i])) for i=1:m]...)
    end

    @constraint(model, cminus,
                1 + sum(φ[i] * p[i] for i=1:m) in SOSCone(),
                domain=blasso.domain, maxdegree=maxdeg)

    @constraint(model, cplus,
                1 - sum(φ[i] * p[i] for i=1:m) in SOSCone(),
                domain=blasso.domain, maxdegree=maxdeg)

    if blasso.λ == 0  # noiseless formulation
        @objective(model, Max, sum(y[i] * p[i] for i=1:m))
    else  # regularized formulation
        # workaround to avoid MosekTools error with quadratic objective
        @variable(model, t)
        @constraint(model, sum(p[i]^2 for i=1:m) <= t)

        @objective(model, Max, sum(y[i] * p[i] for i=1:m) - blasso.λ / 2 * t)
    end

    optimize!(model)
    blasso.objvalue = objective_value(model)
    blasso.dualsol = [value(p[i]) for i=1:m]

    # finds a primal solution given the dual solution
    νplus = moment_matrix(cplus)
    μplus = extractatoms(νplus, ϵ)

    νminus = moment_matrix(cminus)
    μminus = extractatoms(νminus, ϵ)

    try
        atoms = μplus.atoms
        append!(atoms, [WeightedDiracMeasure(atom.center, -atom.weight)
                        for atom in μminus.atoms])
        blasso.μ = AtomicMeasure(μplus.variables, atoms)
    catch error
        if isa(error, ErrorException)
            println("Atom extraction failed")
            blasso.μ = nothing
        end
    end
end

# utility function allowing to setup an example
function example(x::Vector{VT}, locations::Vector{Vector{LT}},
                 amplitudes::Vector{AT}, domain::AbstractSemialgebraicSet,
                 d::Integer, ϵ::Real, σ::Real=0, λ::Real=0) where {VT<:DP.AbstractVariable,
                                                                   LT<:Real,
                                                                   AT<:Real}

    # sought-after measure
    μ0 = AtomicMeasure(x, [WeightedDiracMeasure(locations[i], amplitudes[i])
                           for i=1:length(locations)])

    # measurement functions
    monos = monomials(x, 0:d)
    φ = polynomial.(monos)
    m = length(φ)

    # measurements computation
    y = [expectation(μ0, φ[i]) for i=1:m]
    noisy_y = y + rand(Normal(0, σ * sqrt(mean(y.^2))), m)

    # call to solver
    prob = blasso(y, φ, λ, domain)
    solve!(prob, ϵ)

    # computation of the dual certificate
    dualcertif_poly = sum(prob.dualsol[i] * φ[i] for i=1:m)
    dualcertif(u) = dualcertif_poly(x=>u)

    prob.objvalue, prob.μ, dualcertif
end
