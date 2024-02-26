import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def armijo (f,varVec,df,p,ak=1,c=0.1,rho=0.75):
    while f(varVec+(ak*p)) - f(varVec) > ak*c*(np.dot(df,p)):
        ak *= rho
    return ak

def geraHessiana (f,varVec,h=0.00001):
    H = []
    for i in range(len(varVec)):
        line = []
        for j in range(len(varVec)):
           # print(varVec)
            if(i!=j):
                varTemp = varVec.copy()
                varTemp[i] += h
                varTemp[j] += h
                print(varTemp)
                p1 = f(varTemp)

                varTemp = varVec.copy()
                varTemp[i] += h
                varTemp[j] -= h
                print(varTemp)
                p2 = f(varTemp)

                varTemp = varVec.copy()
                varTemp[i] -= h
                varTemp[j] += h
                print(varTemp)
                p3 = f(varTemp)

                varTemp = varVec.copy()
                varTemp[i] -= h
                varTemp[j] -= h
                print(varTemp)
                p4 = f(varTemp)
                line.append((p1 - p2 - p3 + p4)/((4*h)**2))
            else:
                varTemp = varVec.copy()
                varTemp[i] += h
                print(varTemp)
                p1 = f(varTemp)

                varTemp = varVec.copy()
                print(varTemp)
                p2 = f(varTemp)

                varTemp = varVec.copy()
                varTemp[i] -= h
                print(varTemp)
                p3 = f(varTemp)

                line.append((p1 - 2*p2 + p3)/(h**2))

        H.append(np.array(line))
    return np.array(H)
def geraGrad (f,varVec,h=0.00001):
    df = []
    for i in range(len(varVec)):
        varTemp = varVec
        varTemp[i] +=  h
        p1 = f(varTemp)
        varTemp[i] -= 2*h
        p2 = f(varTemp)
        df.append((p1-p2)/(2*h))
    return np.array(df)

def metodoGrad (f,x0,eps):
    pts = [x0]
    x = x0
    err = 1000
    while err > eps:
        grad = geraGrad(f,x)
        p = -grad
        print("grad: ",grad)
        ak = armijo(f,x,grad,p)
        print(ak)
        xk = x + ak*p
        print("x xk",x,xk)
        err = np.max(abs(xk-x))
        print("err: ", err)
        x = xk
        pts.append(xk)
    return pts

def metodoNewton (f,x0,eps):
    pts = [x0.copy()]
    x = x0.copy()
    err = 1000
    while err > eps:
        grad = geraGrad(f,x.copy())
        H = geraHessiana(f,x.copy())
        print("H: ",H)
        p = -np.linalg.inv(H) @ grad
        print("p: ", p)
        ak = armijo(f,x,grad,p)
        print("ak: ", ak)
        xk = x + ak*p
        err = np.max(abs(xk-x))
        print("err: ", err)
        x = xk.copy()
        pts.append(xk)
    return pts

def metodoNewtonMod (f,x0,eps0=0.001,eps1=0.0001):
    pts = [x0.copy()]
    x = x0.copy()
    err = 1000
    while err > eps0:
        grad = geraGrad(f,x)
        H = geraHessiana(f,x)
        print("H: ",H)
        p = -np.linalg.inv(H+(eps1*np.eye(H.shape[0]))) @ grad
        print("p: ", p)
        ak = armijo(f,x,grad,p)
        print("ak: ", ak)
        xk = x + ak*p
        err = np.max(abs(xk-x))
        print("err: ", err)
        x = xk
        pts.append(xk)
    return pts

def metodoGradConj (x0):

    return 0

### Himmelblau
def funcHimmelblau(x):
    f = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2
    return f

def plotHimmelblau():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x,y = np.mgrid[-5:5:30j,-5:5:30j]
    z = funcHimmelblau([x,y])   
    ax.plot_surface(x,y,z, cmap=cm.hot, ec='k',alpha=0.5)#,rstride=1,cstride=1)
    
    x0 = np.array([0,0]).astype(np.float32)
    pts = metodoNewton(funcHimmelblau,x0,eps=0.001)
    print("pts:", pts)
    for i in pts:
        ax.scatter(i[0],i[1],funcHimmelblau(i),color='green', s=50)
        #ax.plot3D(i[0],i[1],funcHimmelblau(i),color='green')
    #ax.view_init(30,30,0)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    plt.show()
plotHimmelblau()

