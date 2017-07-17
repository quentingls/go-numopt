package newton

import (
	"errors"
	"gonum.org/v1/gonum/mat"
)

// TwiceDifferenciableFunction should implement grandient and henssian in order
// to create a second degree Taylor expasion.
type TwiceDifferenciableFunction interface {
	ValueAt(x *mat.Vector) float64
	GradientAt(x *mat.Vector) *mat.Vector
	HenssianAt(x *mat.Vector) *mat.Dense
}

// OptimisationParameters contains the parameters defining the behaviour of the
// optimisation iteration. X0 is the starting point, Alpha is the learning rate,
// Epsilon the targeted accuracy and N the maximum iteration.
type OptimisationParameters struct {
	X0      *mat.Vector
	Alpha   float64
	Epsilon float64
	N       int
}

// NewtonRaphsonOptimise implements the Newton Raphson Optimisation algorithm
// on the twice-differentiale function f.
func NewtonRaphsonOptimise(f TwiceDifferenciableFunction, param OptimisationParameters) (*mat.Vector, error) {
	x := param.X0
	for i := 0; i < param.N; i++ {
		henssian, gradient := f.HenssianAt(x), f.GradientAt(x)
		r, _ := henssian.Dims()
		invertHenssian := mat.NewDense(r, r, nil)
		invertHenssian.Inverse(henssian)
		prod := mat.NewDense(r, 1, nil)
		prod.Mul(invertHenssian, gradient)
		d := prod.ColView(0)
		d.ScaleVec(-param.Alpha, d)
		x.SubVec(x, d)
		if f.ValueAt(x) < param.Epsilon {
			return x, nil
		}
	}
	return x, errors.New("maximum number of iteraton has been reached")
}

// TODO: Implement backtracking for step size optimisation
func backtracking() {}
