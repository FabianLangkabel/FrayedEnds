import numpy
import scipy

class LegendreBasisFunction:

    def __init__(self, k, *args, **kwargs):
        self.k = k

    def __call__(self, x, *args, **kwargs):
        # shift to [0,1] interval
        x = 2*x-1
        # normalize on [-1,1]
        N = numpy.sqrt((2.0*self.k+1.0)/2.0)
        # normalize shifted function on [0,1]
        N*= numpy.sqrt(2.0)
        return N*scipy.special.legendre(n=self.k)(x)


def integrate(f,order,a=0,b=1, verbose=False):
    """
    integrate in interval [a,b] subset to [0,1]
    """
    x, w = scipy.special.roots_legendre(n=order+1)
    # shift points from [-1,1] to [0,1]
    #x = (x+1)/2
    # shift points form [0,1] to [a,b]
    #x = x*(b-a) + a
    
    # change of interval
    x = (b-a)/2*x + (a+b)/2
    N = (b-a)/2
    y = N*f(x)
    return numpy.sum(w*y)

# convenient for inner product integrals
def inner(f,g,order,a=0,b=1):
    return integrate(lambda x: f(x)*g(x), order=order, a=a, b=b)


class ProjectedFunction:
  
    def rescale(self, f):
        n = self.n
        l = self.l
        
        @numpy.vectorize
        def window(f,x):
            if x<0.0 or x>1.0:
                return 0.0
            else:
                return f(x)

        N = 2**(0.5*n)
        return lambda x: N*window(f,2**n*x-l)

    def __init__(self, f, k, n=0, l=0, *args, **kwargs):
        assert k>0
        assert l<2**n
        self.k = k
        self.n = n
        self.l = l

        # initialize basis functions
        self.basis = [self.rescale(LegendreBasisFunction(k=kk)) for kk in range(k)]
        self.a = 1/(2**n)*l
        self.b = 1/(2**n)*(l+1)
        self.coeffs = [inner(f,self.basis[n],order=k, a=self.a, b=self.b) for n in range(k)]

    def __call__(self, x, *args, **kwargs):
        y = [self.coeffs[n]*self.basis[n](x) for n in range(self.k)]
        return sum(y)
    
def f(x):
    x = 2*x-1
    a= numpy.exp(-10*(x+0.5)**2)
    b= numpy.sinc(x)
    return numpy.sin(numpy.exp(-3.0*(x)))*b

import matplotlib.pyplot as plt
x = numpy.linspace(0.0,1.0,500)
fx = f(x)

F = ProjectedFunction(f, k=5)

plt.plot(x,fx, color="black", label="f(x)")
plt.legend()
plt.savefig("f.pdf")

color = ["forestgreen", "tab:blue", "tab:orange", "tab:red", "navy"]
for i,k in enumerate([5,10,20,40]):
    F = ProjectedFunction(f, k=k)
    plt.plot(x,fx, label="f(x)", color="black")
    plt.plot(x,F(x), color=color[i], linestyle="--", label="k={}".format(k), linewidth=2)
    plt.legend()
    plt.savefig("f_standard_{}.pdf".format(k))
    plt.show()
