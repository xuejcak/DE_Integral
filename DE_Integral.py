import numpy as np
'''
[parameters]
    f    : integrand f(x)
    a    : lower limit of integration
    b    : upper limit of integration
    eps  : relative error requested
    i    : approximation to the integral
    err  : estimate of the absolute error
[remarks]
    function
        f(x) needs to be analytic over (a,b).
    relative error
        eps is relative error requested excluding cancellation of 
        significant digits.
        i.e. eps means : (absolute error) / (integral_a^b |f(x)| dx).
        eps does not mean: (absolute error) / I.
    error message
        err >= 0: normal termination.
        err < 0: abnormal termination (m >= mmax).
            i.e. convergent error is detected :
                1. f(x) or (d/dx)^n f(x) has discontinuous points or
                  sharp peaks over (a,b).
                  you must divide the interval (a,b) at this points.
                2. relative error of f(x) is greater than eps.
                3. f(x) has oscillatory factor and frequency of the
                  oscillation is very high.
'''
def intde(f, a, b, eps):
    mmax = 256
    efs, hoff = 0.1, 8.5
    pi2 = np.pi/2.
    epsln = 1-np.log(efs*eps)
    epsh = np.sqrt(efs*eps)
    h0 = hoff/epsln
    ehp = np.exp(h0)
    ehm = 1/ehp
    epst = np.exp(-ehm*epsln)
    ba = b-a
    ir = f((a+b)*0.5)*(ba*0.25)
    i = ir*(2*pi2);
    err = np.abs(i)*epst
    h = 2*h0
    m = 1
    
    condition0 = True
    while condition0:
        iback = i
        irback = ir
        t = h*0.5
        condition1 = True
        while condition1:
            em = np.exp(t)
            ep = pi2*em
            em = pi2/em
            condition2 = True
            while condition2:
                xw = 1/(1+np.exp(ep-em))
                xa = ba*xw
                wg = xa*(1-xw)
                fa = f(a+xa)*wg
                fb = f(b-xa)*wg
                ir += fa+fb
                i += (fa+fb)*(ep+em)
                errt = (np.abs(fa)+np.abs(fb))*(ep+em)
                if m == 1:
                    err += errt*epst
                ep *= ehp
                em *= ehm
                condition2 = errt > err or xw > epsh
            t += h
            condition1 = t < h0
        if m == 1:
            errh = (err/epst)*epsh*h0
            errd = 1+2*errh
        else:
            errd = h*(np.abs(i-2*iback) + 4*np.abs(ir-2*irback))
        h *= 0.5
        m *= 2
        condition0 = errd > errh and m < mmax
    i *= h
    if errd > errh:
        err = -errd*m
    else:
        err = errh*epsh*m/(2*efs)
    
    result = {"integral": i, "error": err}
    return result

def intde2ini(lenaw, tiny, eps, aw):
    # adjustable parameter
    efs, hoff = 0.1, 8.5
    # --------------------
    pi2 = np.pi*0.5
    tinyln = -np.log(tiny)
    epsln = 1-np.log(efs*eps)
    h0 = hoff/epsln
    ehp = np.exp(h0)
    ehm = 1/ehp
    aw[2] = eps
    aw[3] = np.exp(-ehm*epsln)
    aw[4] = np.sqrt(efs*eps)
    noff = 5
    aw[noff] = 0.5
    aw[noff+1] = h0
    aw[noff+2] = pi2*h0*0.5
    h = 2
    nk = 0
    k = noff+3
    condition0 = True
    while condition0:
        t = h*0.5
        condition1 = True
        while condition1:
            em = np.exp(h0*t)
            ep = pi2*em
            em = pi2/em
            j = k
            condition2 = True
            while condition2:
                xw = 1/(1+np.exp(ep-em))
                wg = xw*(1-xw)*h0
                aw[j] = xw
                aw[j+1] = wg*4
                aw[j+2] = wg*(ep+em)
                ep *= ehp
                em *= ehm
                j += 3
                condition2 = ep < tinyln and j <= lenaw-3
            t += h
            k += nk
            condition1 = t < 1
        h *= 0.5
        if nk == 0:
            if j > lenaw-6: j -= 3
            nk = j-noff
            k += nk
            aw[1] = nk
        condition0 = 2*k-noff-3 <= lenaw
    aw[0] = k-3

def intde2(f, a, b, aw):
    noff = 5
    lenawm = int(aw[0]+0.5)
    nk = int(aw[1]+0.5)
    epsh = aw[4]
    ba = b-a
    i = f((a+b)*aw[noff])
    ir = i*aw[noff+1]
    i *= aw[noff+2]
    err = np.abs(i)
    k = nk+noff
    j = noff
    condition0 = True
    while condition0:
        j += 3
        xa = ba*aw[j]
        fa = f(a+xa)
        fb = f(b-xa)
        ir += (fa+fb)*aw[j+1]
        fa *= aw[j+2]
        fb *= aw[j+2]
        i += fa+fb
        err += np.abs(fa)+np.abs(fb)
        condition0 = aw[j] > epsh and j < k
    errt = err*aw[3]
    errh = err*epsh
    errd = 1+2*errh
    jtmp = j
    
    while np.abs(fa) > errt and j < k:
        j += 3
        fa = f(a+ba*aw[j])
        ir += fa*aw[j+1]
        fa *= aw[j+2]
        i += fa
    jm = j
    j = jtmp
    while np.abs(fb) > errt and j < k:
        j += 3
        fb = f(b-ba*aw[j])
        ir += fb*aw[j+1]
        fb *= aw[j+2]
        i += fb
    if j<jm: jm=j
    jm -= noff+3
    h = 1
    m = 1
    klim = k+nk
    while errd > errh and klim <= lenawm:
        iback = i
        irback = ir
        condition0 = True
        while condition0:
            jtmp = k+jm
            for j in range(k+3, jtmp+1, 3):
                xa = ba*aw[j]
                fa = f(a+xa)
                fb = f(b-xa)
                ir += (fa+fb)*aw[j+1]
                i += (fa+fb)*aw[j+2]
            k += nk
            j = jtmp
            condition1 = True
            while condition1:
                j += 3
                fa = f(a+ba*aw[j])
                ir += fa*aw[j+1]
                fa *= aw[j+2]
                i += fa
                condition1 = np.abs(fa) > errt and j < k
            j = jtmp
            condition1 = True
            while condition1:
                j += 3
                fb = f(b-ba*aw[j])
                ir += fb*aw[j+1]
                fb *= aw[j+2]
                i += fb
                condition1 = np.abs(fb) > errt and j < k
            condition0 = k < klim
            errd = h*(np.abs(i-2*iback)+np.abs(ir-2*irback))
            h *= 0.5
            m *= 2
            klim = 2*klim - noff
        i *= h*ba
        if errd > errh:
            err = -errd*(m*np.abs(ba))
        else:
            err = err*aw[2]*(m*np.abs(ba))
    result = {"integral": i, "error": err}
    return result    

# test
def f1(x):
    global nfunc
    nfunc += 1
    return 1/np.sqrt(x)
def f2(x):
    global nfunc
    nfunc += 1
    return np.sqrt(4-x*x)
