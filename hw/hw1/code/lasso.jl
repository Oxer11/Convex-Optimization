using CSV, Images, ImageView
using Convex, SCS


function read_image(filename :: String)
    image = CSV.read(filename;header=false)
    Y = convert(Matrix, image)
    return Y
end

function lasso(Y :: Matrix, λ=1.0, p=1)
    
    solver = SCSSolver(verbose=false)
    n, m = size(Y)
    Θ = Variable(n, m)
    
    obj = 0.5 * square(norm(vec(Y - Θ)))
    for i in 1:n
        for j in 1:m
            (j < m) ? (vec0 = abs(Θ[i, j] - Θ[i, j+1])) : (vec0 = 0)
            (i < n) ? (vec1 = abs(Θ[i, j] - Θ[i+1, j])) : (vec1 = 0)
            obj = obj + λ * norm(vcat(vec0, vec1), p)
        end
    end
    
    prob = minimize(obj)
    solve!(prob, solver)
    return prob.optval, Θ.value
end

toy = read_image("../toy.csv")
img = Gray.(toy)
imshow(img)

# Problem 1
opt1, img1 = lasso(toy, 1, 1)
println(opt1)
imshow(Gray.(img1))

# Problem 2
opt2, img2 = lasso(toy, 1, 2)
println(opt2)
imshow(Gray.(img2))

# Problem 3
baboon = read_image("../baboon.csv")
img = Gray.(baboon)
imshow(img)

Θ = []
for p in 1:2
    for λ in 0:8
        opt, img = lasso(baboon, 10^(-λ/4), p)
        push!(Θ, (opt, img))
        println("For $p-norm lasso problem with λ=$(10^(-λ/4)), the optimal value is $opt and the solution image is shown above.")
        imshow(Gray.(img))
    end
end
