using CSV, Images, Plots, ImageView
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
println("Displaying the original image.")
imshow(img)

opt1, img1 = lasso(toy, 1, 1)
println(opt1)
histogram(reshape(img1, length(img1), 1), bins=range(0, 1, length=100))
println("Displaying the solution image.")
imshow(Gray.(img1))

