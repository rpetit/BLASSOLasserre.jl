module BLASSOLasserre

using DynamicPolynomials
using SemialgebraicSets
using SumOfSquares
using MosekTools  # to use a different solver, see SumOfSquares.jl documentation

using Random
using Distributions

import MultivariateMoments.expectation

const DP=DynamicPolynomials

include("blasso.jl")

end
