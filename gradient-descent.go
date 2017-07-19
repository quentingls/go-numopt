package numopt

import (
	"gonum.org/v1/gonum/mat"
)

type GradientDescentOption struct {
	X0      *mat.Vector
	Alpha   float64
	Epsilon float64
	N       int
	F       DifferenciableFuntion
}

func GradientDescentOptimise(opt GradientDescentOption) (*mat.Vector, error) {
	dim, _ := opt.X0.Dims()
	approx := newIdentityApproximation(dim)
	quasiNewtonOpts := QuasiNewtonOptions{
		X0:            opt.X0,
		Alpha:         opt.Alpha,
		Epsilon:       opt.Epsilon,
		N:             opt.N,
		F:             opt.F,
		Approximation: approx,
		H0:            approx.identity,
	}
	return QuasiNewtonOptimise(quasiNewtonOpts)
}

type identityApproximation struct {
	identity *mat.SymBandDense
}

func newIdentityApproximation(dim int) *identityApproximation {
	values := make([]float64, dim)
	for i := 0; i < dim; i++ {
		values[i] = 1
	}
	identity := mat.NewDiagonal(dim, values)
	return &identityApproximation{
		identity,
	}
}

func (approx *identityApproximation) UpdateHenssian(henssian mat.Matrix, deltaGrad *mat.Vector, deltaX *mat.Vector) mat.Matrix {
	return approx.identity
}
