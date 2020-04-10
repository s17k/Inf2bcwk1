import numpy
import scipy.io
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [4, 4]


def task2_genNeuron(W, X, applied_fun):
    # adding a column of ones 
    X_o = numpy.c_[numpy.ones(len(X)), X]
    
    Y = X_o @ W
    
    # finally, return Y with whatever function g is applied
    return numpy.apply_along_axis(applied_fun, 0, Y)

def task2_hNeuron(W, X):    
    def step(x):
        return 1 * (x > 0)
    return task2_genNeuron(W, X, step)

def task2_sNeuron(W, X):
    def g(x):
        return 1/(1+numpy.exp(numpy.longdouble(-x)))
    return task2_genNeuron(W, X, g)


# converts a list to a column vector ndarray
def listToColV(li):
    return numpy.asarray(li).reshape(-1,1)

def task2_hNN_A(X):
    first_layer = [[1.0, 0.9541936586843557, -0.85493703875096],
                   [-1.0, 0.2433319217436556, 0.1032862159048711],
                   [-1.0, -0.14275094489266318, 0.4274526978329043], 
                   [1.0, -0.265467958624089, -0.061666696980078]]
    upper_neuron = [-0.8, 0.25, 0.25, 0.25, 0.25]
    
   
    first_layer_results = [task2_hNeuron(listToColV(W), X) for W in first_layer]
    first_layer_X = numpy.column_stack(first_layer_results)
    final_result = task2_hNeuron(listToColV(upper_neuron), first_layer_X)
    return final_result



# task 2.5
def task2_plot_regions_hNN_A():
    X_N, Y_N = 1000, 1000
    x = numpy.linspace(2.4, 3.1, X_N)
    y = numpy.linspace(3.0, 4.5, Y_N)

    X, Y = numpy.meshgrid(x, y)

    X_grid = []
    for i in range(Y_N):
        for j in range(X_N):
            X_grid.append([X.item(i,j), Y.item(i,j)])

    X_grid = numpy.asarray(X_grid)
    Y_grid = task2_hNN_A(X_grid)

    Z = Y_grid.reshape((Y_N, X_N))
        
    plt.contourf(X, Y, Z)
    plt.savefig('t2_regions_hNN_A.pdf')



def task2_hNN_AB(X):
    first_layer = [
                    [1.0, 0.9541936586843557, -0.85493703875096], 
                    [-1.0, 0.2433319217436556, 0.1032862159048711], 
                    [-1.0, -0.14275094489266318, 0.4274526978329043], 
                    [1.0, -0.265467958624089, -0.061666696980078],
                    [1.0, -0.19159285719085067, -0.04076000581289526], 
                    [-1.0, 0.24689693509583485, 0.2027403058246619], 
                    [-1.0, 0.1242531034272138, 0.2993280530205454], 
                    [1.0, 0.09236618495168845, -0.22632245296746664]
    ]
    
    second_layer = [
                    [-0.8, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0], 
                    [-(1/2+1/16), 0, 0, 0, 0, 1/4, 1/8, 1/8, 1/4]
    ]
    
    final_neuron = [-0.5, -1.0, 1.0]
    
    first_layer_results = [task2_hNeuron(listToColV(W), X) for W in first_layer]
    first_layer_X = numpy.column_stack(first_layer_results)
    
    second_layer_results = [task2_hNeuron(listToColV(W), first_layer_X) for W in second_layer]
    second_layer_X = numpy.column_stack(second_layer_results)
    
    final_result = task2_hNeuron(listToColV(final_neuron), second_layer_X)
    
    return final_result


def task2_plot_regions_hNN_AB():
    X_N, Y_N = 1000, 1000
    x = numpy.linspace(-2, 7, X_N)
    y = numpy.linspace(-2, 7, Y_N)

    X, Y = numpy.meshgrid(x, y)

    X_grid = []
    for i in range(Y_N):
        for j in range(X_N):
            X_grid.append([X.item(i,j), Y.item(i,j)])

    X_grid = numpy.asarray(X_grid)
    Y_grid = task2_hNN_AB(X_grid)

    Z = Y_grid.reshape((Y_N, X_N))
            
    plt.contourf(X, Y, Z)
    plt.savefig('t2_regions_hNN_AB.pdf')


def task2_sNN_AB(X):
    first_layer = [
                    [1.0, 0.9541936586843557, -0.85493703875096], 
                    [-1.0, 0.2433319217436556, 0.1032862159048711], 
                    [-1.0, -0.14275094489266318, 0.4274526978329043], 
                    [1.0, -0.265467958624089, -0.061666696980078],
                    [1.0, -0.19159285719085067, -0.04076000581289526], 
                    [-1.0, 0.24689693509583485, 0.2027403058246619], 
                    [-1.0, 0.1242531034272138, 0.2993280530205454], 
                    [1.0, 0.09236618495168845, -0.22632245296746664]
    ]
    
    def make_big(vec):
        return (numpy.asarray(vec)*1000.0).tolist()
    
    first_layer = [make_big(neuron) for neuron in first_layer]
    
    
    second_layer = [
                    make_big([-0.8, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0]), 
                    make_big([-(1/2+1/16), 0, 0, 0, 0, 1/4, 1/8, 1/8, 1/4]),
    ]
    
    final_neuron = make_big([-0.5, -1.0, 1.0])
    
    first_layer_results = [task2_sNeuron(listToColV(W), X) for W in first_layer]
    first_layer_X = numpy.column_stack(first_layer_results)
    
    second_layer_results = [task2_sNeuron(listToColV(W), first_layer_X) for W in second_layer]
    second_layer_X = numpy.column_stack(second_layer_results)
    
    final_result = task2_sNeuron(listToColV(final_neuron), second_layer_X)
    
    return final_result


def task2_plot_regions_sNN_AB():
    X_N, Y_N = 1000, 1000
    x = numpy.linspace(-2, 7, X_N)
    y = numpy.linspace(-2, 7, Y_N)

    X, Y = numpy.meshgrid(x, y)

    X_grid = []
    for i in range(Y_N):
        for j in range(X_N):
            X_grid.append([X.item(i,j), Y.item(i,j)])

    X_grid = numpy.asarray(X_grid)
    Y_grid = task2_sNN_AB(X_grid)

    Z = Y_grid.reshape((Y_N, X_N))
            
    plt.contourf(X, Y, Z)
    plt.savefig('t2_regions_sNN_AB.pdf')
