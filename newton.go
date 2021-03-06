package numopt

import (
	"errors"
	"gonum.org/v1/gonum/mat"
	"math"
)

// DifferenciableFuntion represents a differenciable
// function.
type DifferenciableFuntion interface {
	ValueAt(x *mat.Vector) float64
	GradientAt(x *mat.Vector) *mat.Vector
}

// TwiceDifferenciableFunction represents a twice differenciable
// function.
type TwiceDifferenciableFunction interface {
	DifferenciableFuntion
	HenssianAt(x *mat.Vector) mat.Matrix
}

// NewtonQuasiUpdate allows for updating invert henssian
// by approxiation.
type NewtonQuasiUpdate func(henssian mat.Matrix, deltaGrad *mat.Vector, deltaX *mat.Vector) mat.Matrix

// QuasiNewtonOptions contains the parameters defining the behaviour of the
// optimisation iteration. X0 is the starting point, H0 is the approximation
// of inverse of the henssian at X0, Alpha is the learning rate,
// Epsilon the targeted accuracy, N the maximum iteration
// and F the differenciable function.
type QuasiNewtonOptions struct {
	X0          *mat.Vector
	H0          mat.Matrix
	Alpha       float64
	Epsilon     float64
	N           int
	F           DifferenciableFuntion
	QuasiUpdate NewtonQuasiUpdate
}

// QuasiNewtonOptimise implements the quasi newton methond. Here we avoid
// computing the invert of the henssian by using an approximation.
func QuasiNewtonOptimise(opt QuasiNewtonOptions) (*mat.Vector, error) {
	x, invertHenssian := opt.X0, opt.H0
	valueAtX, valueAtNextX := opt.F.ValueAt(x), 0.0
	delta := 0.0
	dim, _ := x.Dims()
	direction := mat.NewVector(dim, nil)
	nextX, nextGradient := mat.NewVector(dim, nil), mat.NewVector(dim, nil)
	deltaX, deltaGradient := mat.NewVector(dim, nil), mat.NewVector(dim, nil)
	for i := 0; i < opt.N; i++ {
		gradient := opt.F.GradientAt(x)
		updateDirection(
			direction,
			invertHenssian,
			gradient,
			opt.Alpha,
		)
		nextX.SubVec(x, direction)
		valueAtNextX = opt.F.ValueAt(nextX)
		delta = (valueAtX - valueAtNextX) / valueAtNextX
		if math.Abs(delta) < opt.Epsilon {
			return nextX, nil
		}
		nextGradient = opt.F.GradientAt(x)
		deltaX.SubVec(nextX, x)
		deltaGradient.SubVec(nextGradient, gradient)
		invertHenssian = opt.QuasiUpdate(
			invertHenssian,
			deltaGradient,
			deltaX,
		)
		x = nextX
		valueAtX = valueAtNextX
	}
	return x, errors.New("maximum number of iteraton has been reached")
}

// NewtonRaphsonOptions contains the parameters defining the behaviour of the
// optimisation iteration. X0 is the starting point, Alpha is the learning rate,
// Epsilon the targeted accuracy, N the maximum iteration and F which is the
// Twice differenciable function on which the optimisation is ran.
type NewtonRaphsonOptions struct {
	X0      *mat.Vector
	Alpha   float64
	Epsilon float64
	N       int
	F       TwiceDifferenciableFunction
}

// NewtonRaphsonOptimise implements the Newton Raphson Optimisation algorithm
// on the twice-differentiale function f.
func NewtonRaphsonOptimise(opt NewtonRaphsonOptions) (*mat.Vector, error) {
	x := opt.X0
	valueAtX, valueAtNextX := opt.F.ValueAt(x), 0.0
	delta := 0.0
	dim, _ := x.Dims()
	direction := mat.NewVector(dim, nil)
	invertHenssian := mat.NewDense(dim, dim, nil)
	for i := 0; i < opt.N; i++ {
		invertHenssian.Inverse(opt.F.HenssianAt(x))
		updateDirection(
			direction,
			invertHenssian,
			opt.F.GradientAt(x),
			opt.Alpha,
		)
		x.SubVec(x, direction)
		valueAtNextX = opt.F.ValueAt(x)
		delta = (valueAtX - valueAtNextX) / valueAtNextX
		if math.Abs(delta) < opt.Epsilon {
			return x, nil
		}
		valueAtX = valueAtNextX
	}
	return x, errors.New("maximum number of iteraton has been reached")
}

// Calculates the iteration direction.
func updateDirection(dir *mat.Vector, invertHenssian mat.Matrix, gradient *mat.Vector, alpha float64) {
	r, _ := invertHenssian.Dims()
	prod := mat.NewDense(r, 1, nil)
	prod.Mul(invertHenssian, gradient)
	*dir = *prod.ColView(0)
	dir.ScaleVec(alpha, dir)
}

type BacktrackingOption struct {
	Tau float64
}

// TODO: Implement backtracking for step size optimisation
func backtracking(x, d *mat.Vector) float64 {
	return 0
}
