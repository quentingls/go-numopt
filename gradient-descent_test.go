package numopt

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

// f(x, y) = x*x - 4
// grad(f)(x,y) = 2*x
func (f function2) ValueAt(x *mat.Vector) float64 {
	x1, x2 := x.At(0, 0), x.At(1, 0)
	return x1*x1 + x2*x2
}

func (f function2) GradientAt(x *mat.Vector) *mat.Vector {
	x1, x2 := x.At(0, 0), x.At(1, 0)
	value := []float64{2 * x1, 2 * x2}
	return mat.NewVector(2, value)
}

func (f function2) HenssianAt(x *mat.Vector) mat.Matrix {
	value := []float64{
		2, 0,
		0, 2,
	}
	return mat.NewDense(2, 2, value)
}

type function2 struct{}

func TestGradientDescent(t *testing.T) {
	x0Value := []float64{10, 10}
	param := GradientDescentOption{
		X0:      mat.NewVector(2, x0Value),
		Alpha:   0.1,
		Epsilon: 0.01,
		N:       100,
		F:       function2{},
	}
	res, err := GradientDescentOptimise(param)
	if err != nil {
		t.Errorf("should have nil error, having: %s", err.Error())
	}
	if math.Abs(param.F.ValueAt(res)) >= math.Abs(param.Epsilon) {
		t.Errorf("iteration does not match accuracy requirement")
	}
}