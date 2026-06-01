# Porownanie GEMM w Conv: alokujace * vs mul! vs BLAS.gemm!
include("src/MiniAD.jl")
using .MiniAD
using LinearAlgebra
using LinearAlgebra.BLAS: gemm!
using Random
using Printf

Random.seed!(0)

function bench_gemm_variants(; reps=500)
    K, C_out, P = 27, 6, 169  # pierwsza warstwa Conv (3x3x1 -> 6, mapa 28x28)
    Wk = rand(Float32, K, C_out)
    x_col = rand(Float32, K, P)
    y_alloc = zeros(Float32, C_out, P)
    y_mul = zeros(Float32, C_out, P)
    y_gemm = zeros(Float32, C_out, P)

    t_alloc = @elapsed for _ in 1:reps
        y_alloc .= transpose(Wk) * x_col
    end
    t_mul = @elapsed for _ in 1:reps
        mul!(y_mul, transpose(Wk), x_col)
    end
    t_gemm = @elapsed for _ in 1:reps
        gemm!('T', 'N', 1.0f0, Wk, x_col, 0.0f0, y_gemm)
    end

  alloc_bytes = @allocated begin
        for _ in 1:reps
            y_alloc .= transpose(Wk) * x_col
        end
    end

    return (;
        reps,
        t_alloc,
        t_mul,
        t_gemm,
        alloc_bytes,
        max_diff_mul_gemm=maximum(abs, y_mul - y_gemm),
    )
end

micro = bench_gemm_variants()
println("=== Micro GEMM (K=27, C_out=6, P=169, reps=$(micro.reps)) ===")
@printf("allocating Wk'*x_col : %.4fs  (@allocated %d reps = %d B)\n", micro.t_alloc, micro.reps, micro.alloc_bytes)
@printf("mul!(y, Wk', x)       : %.4fs  (%.2fx vs alloc)\n", micro.t_mul, micro.t_alloc / micro.t_mul)
@printf("BLAS.gemm!            : %.4fs  (%.2fx vs alloc)\n", micro.t_gemm, micro.t_alloc / micro.t_gemm)
@printf("max |mul! - gemm!|    : %.3e\n", micro.max_diff_mul_gemm)

println("\n(Pelny run: julia --project=. run.jl oraz benchmark_profile.jl)")
