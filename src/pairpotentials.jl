



function LennardJones(
    emin::Unitful.Energy,
    rmin::Unitful.Length,
    id1, id2,
    cutoff::Unitful.Length;
    parametric=false
)
    @assert emin < 0u"eV"
    @assert rmin > 0u"pm"
    @assert cutoff > rmin
    ε = -ustrip(emin)
    σ = ustrip(unit(cutoff), rmin) / 2^(1//6)
    A = 4ε * σ^12
    B = 4ε * σ^6

    if parametric
        return ParametricPairPotential(
            (r,c) -> c[1]/r^12 - c[2]/r^6,
            SVector(A, B),
            (id1, id2),
            cutoff,
            zero(emin)
        )
    else
        return SimplePairPotential(
            r -> A/r^12 - B/r^6,
            (id1, id2),
            cutoff,
            zero(emin)
        )
    end
end
