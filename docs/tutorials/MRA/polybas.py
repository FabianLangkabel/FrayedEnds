import numpy
import scipy

class LegendreBasisFunction:
    
    def __init__(self, k, *args, **kwargs):
        self.k = k

    def __call__(self, x, *args, **kwargs):
        # normalize on [-1,1] interval
        N = numpy.sqrt((2.0*self.k+1.0)/2.0)
        # shift to [0,1] interval
        x = 2*x-1
        return N*scipy.special.legendre(n=self.k)(x)

def integrate(f,order):
    """
    integrate in interval [0,1]
    """
    x, w = scipy.special.roots_legendre(n=order+1)
    # shift points from [-1,1] to [0,1]
    x = (x+1)/2
    y = f(x)
    return numpy.sum(w*y)

class Box:

    def __init__(self, f, k=1):
        if k<1:
            raise Exception("order must be larger than zero")
        
        self.k = k
        coeffs = []
        
        # initialize basis functions
        self.basis = [LegendreBasisFunction(k=kk) for kk in range(k)]

        # compute the coefficients in the legedre basis
        # by computing the integral via Gauss-Legedre Quadrature
        for n in range(k):
            ff = lambda x: self.basis[n](x)*f(x)
            c = integrate(ff,order=k)
            coeffs.append(c)
        self.coeffs = coeffs

    def __call__(self, x, *args, **kwargs):
        y = [self.coeffs[n]*self.basis[n](x) for n in range(self.k)]
        return sum(y)

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    def f(x):
        x = 2*x-1
        a= numpy.exp(-10*(x+0.5)**2) 
        b= numpy.sinc(x)
        return numpy.sin(numpy.exp(-3.0*(x)))*b 
    
    
    box = Box(f, k=5)
    
    x = [x for x in numpy.linspace(0.0,1.0,1000)]
    y1 = [f(xx) for xx in x]
    y2 = [box(xx) for xx in x]
    
    plt.figure()
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.show()

