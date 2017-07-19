package numopt

import (
	"gonum.org/v1/gonum/mat"
	"testing"
)

type function struct {
}

// f(x, y) = x (2x - y) + 2y
// grad(f)(x,y) = [4x - y, -x + 2]
// henssian(f) (x,y) = |4,-1|
// 					   |-1,0|

func (f function) ValueAt(x *mat.Vector) float64 {
	x1, x2 := x.At(0, 0), x.At(1, 0)
	return x1*x1 - 4*x1 + x2*x2 - 10
}

func (f function) GradientAt(x *mat.Vector) *mat.Vector {
	x1, x2 := x.At(0, 0), x.At(1, 0)
	value := []float64{2*x1 - 4, 2 * x2}
	return mat.NewVector(2, value)
}

func (f function) HenssianAt(x *mat.Vector) mat.Matrix {
	value := []float64{
		2, 0,
		0, 2,
	}
	return mat.NewDense(2, 2, value)
}

func TestNewtonRaphson(t *testing.T) {
	x0Value := []float64{10, 10}
	param := NewtonRaphsonOptions{
		X0:      mat.NewVector(2, x0Value),
		Alpha:   0.1,
		Epsilon: 0.0001,
		N:       10000,
		F:       function{},
	}
	res, err := NewtonRaphsonOptimise(param)
	if err != nil {
		t.Errorf("should have nil error, having: %s", err.Error())
	}
	if param.F.ValueAt(res) > -13.5 || param.F.ValueAt(res) < -14.5 {
		t.Errorf("iteration does not match accuracy requirement")
	}
}
