package numopt

import (
	"gonum.org/v1/gonum/mat"
)

// GradientDescentOption defines the optimizaion behaviour
// X0 is the starting point, Alpha the learning rate,
// Epsilon is the accuracy, N the maximum iteration and
// F a single differenciable function.
type GradientDescentOption struct {
	X0      *mat.Vector
	Alpha   float64
	Epsilon float64
	N       int
	F       DifferenciableFuntion
}

// GradientDescentOptimise finds the roots using the quasi newton
// approximation cancelling the effect of the invert Henssian by
// always setting it to the identity matrix.
func GradientDescentOptimise(opt GradientDescentOption) (*mat.Vector, error) {
	dim, _ := opt.X0.Dims()
	update := identityUpdate(dim)
	quasiNewtonOpts := QuasiNewtonOptions{
		X0:          opt.X0,
		Alpha:       opt.Alpha,
		Epsilon:     opt.Epsilon,
		N:           opt.N,
		F:           opt.F,
		QuasiUpdate: update,
		H0:          identity(dim),
	}
	return QuasiNewtonOptimise(quasiNewtonOpts)
}

func identityUpdate(dim int) NewtonQuasiUpdate {
	id := identity(dim)
	return func(henssian mat.Matrix, deltaGrad *mat.Vector, deltaX *mat.Vector) mat.Matrix {
		return id
	}
}

func identity(dim int) mat.Matrix {
	values := make([]float64, dim)
	for i := 0; i < dim; i++ {
		values[i] = 1
	}
	identity := mat.NewDiagonal(dim, values)
	return identity
}
