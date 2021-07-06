import numpy as np
import math
# © Copyright 2008-2020, The SciPy community. Last updated on Nov 04, 2020. Created using Sphinx 3.1.2.
from scipy import stats
# © Copyright 2011–2020, The Astropy Developers. Created using Sphinx 3.2.1. Last built 22 Oct 2020.
import astropy.stats

# D_n+
def dpos(u, n):
    dp = (np.arange(1.0, n+1) / n - u).max()
    return (dp,pvalue(dp,'D_n+'))

# D_n-
def dneg(u, n):
    dn = (u - np.arange(0.0, n)/n).max()
    return (dn,pvalue(dn,'D_n-'))

# Kolmogorov
def ks_test(u,cdf):
    t = stats.kstest(u,cdf)
    return (t.statistic, t.pvalue)
    
def Kolmogorov(u, n):
    s = max(dpos(u,n)[0],dneg(u,n)[0])
    return (s,pvalue(s,'Kolmogorov'))

# Kuiper
def kuiper_test(u,n):
    return astropy.stats.kuiper(u)
    
def Kuiper(u, n):
    s = dpos(u,n)[0] + dneg(u,n)[0]
    return (s,pvalue(s,'Kuiper'))

# Cramér-von Mises    
def CramervonMises(u,n):
    s = sum((u - (2*np.arange(1.0, n+1)-1)/(2*n))**2) + 1/(12*n)
    return (s,pvalue(s,'Cramér-von Mises'))

# Anderson–Darling
def AndersonDarling(u,n):
    a = sum((2*np.arange(1.0, n+1)-1)*(np.log(u) + np.log(1-u[::-1])))
    s = -n - a/n
    return (s,pvalue(s,'Anderson–Darling'))

# Watson 
def Watson(u,n):
    s = CramervonMises(u,n)[0] - n*(sum(u)/n - 0.5)**2
    return (s,pvalue(s,'Watson'))

# C_n+
def cpos(u, n):
    cp = (u - np.arange(1.0, n+1)/(n+1)).max()
    return (cp,pvalue(cp,'C_n+'))
    
# C_n-
def cneg(u, n):
    cn = (-u + np.arange(1.0, n+1)/(n+1)).max()
    return (cn,pvalue(cn,'C_n-'))

# C_n
def c(u, n):
    s = max(cpos(u,n)[0],cneg(u,n)[0])
    return (s,pvalue(s,'C_n'))

# K_n
def k(u, n):
    s = cpos(u,n)[0]+cneg(u,n)[0]
    return (s,pvalue(s,'K_n'))

# T_1
def t1(u, n):
    st1 = sum(abs(u-np.arange(1.0, n+1)/(n+1))/n)
    return (st1,pvalue(st1,'T_1'))
    
# T_2
def t2(u, n):
    st2 = sum(((u-np.arange(1.0, n+1)/(n+1))**2)/n)
    return (st2,pvalue(st2,'T_2'))

# T_1'
def tt1(u, n):
    stt1 = sum(abs(u-np.arange(0.0, n)/(n-1))/n)
    return (stt1,pvalue(stt1,'TT_1'))

# T_2' 
def tt2(u, n):
    stt2 = sum(((u-np.arange(0.0, n)/(n-1))**2)/n)
    return (stt2,pvalue(stt2,'TT_2'))

# Greenwood
def Greenwood(u,n):
    u = np.concatenate([[0],u,[1]])
    s = sum(np.diff(u)**2)
    return (s,pvalue(s,'Greenwood'))

# Quesenberry Miller: same result as Greenwood if there is not i such that u[i] < u[i+1] < u[i+2]
def QuesenberryMiller(u,n):
    u = np.concatenate([[0],u,[1]])
    d = np.diff(u)
    s = sum(d**2) + sum(np.multiply(d[1:],d[:n]))
    return (s,pvalue(s,'Quesenberry Miller')) 

# Cressie and Read 1: test result is nan if l < 0 and (n+1)*u[0] = 0 (u[0] cannot be 0) or (n+1)*d = 0 (u[i] != u[i-1] for all i)
def CressieRead1(u,n,l):
    u = np.concatenate([[0],u,[1]])
    d = np.diff(u)
    cr = sum(d*((((n+1)*d)**l)-1))
    s = (2*n/(l*(l+1)))*cr
    return (s,pvalue(s,'Cressie and Read 1',l))

# Moran: test result is inf if (n+1)*u[0] = 0 (u[0] cannot be 0) or (n+1)*d = 0 (u[i] != u[i-1] for all i)
def Moran(u,n):
    u = np.concatenate([[0],u,[1]])
    s = -2*sum(np.log((n+1)*np.diff(u)))
    return (s,pvalue(s,'Moran'))
 
# Cressie 1: test result is -inf if G_i^(m) = 0 (u[i] != u[i+m] for all i) => Take big values for m
def Cressie1(u,n,m):
    s = np.log(u[m-1]) + sum(np.log(u[m:] - u[:(n-m)])) + np.log(1 - u[n-m])
    return (s,pvalue(s,'Cressie 1'))
    
# Cressie 2
def Cressie2(u,n,m):
    s = (n*u[m-1])**2 + sum((n*(u[m:] - u[:(n-m)]))**2) + (n*(1 - u[n-m]))**2
    return (s,pvalue(s,'Cressie 2'))

# Vasicek: test result is -inf if n*(u[r1]-u[r2])/(2*m) = 0 (u[i+m] != u[i-m] for all i) => Take big values for m
def Vasicek(u,n,m):
    s = (sum(np.log(n*(u[m:2*m+1]-u[0])/(2*m))) + sum(np.log(n*(u[n-1]-u[n-2*m-1:n-m])/(2*m))) + sum(np.log(n*(u[2*m+1:n-1]-u[1:n-2*m-1])/(2*m))))/n
    return (s,pvalue(s,'Vasicek'))
    
# Swartz
def Swartz(u,n):
    u = np.concatenate([[-u[0]],u,[2-u[n-1]]])
    s = n*(sum(((u[2:]-u[:n])/2 - 1/n)**2))/2
    return (s,pvalue(s,'Swartz')) 

# Morales
def phi(l,x):
    if l == 0:
        return x*np.log(x)
    elif l == -1:
        return -np.log(x)
    else:
        return (x**(l+1)-1)/(l*(l+1))
 
def Morales(u,n,m,l):
    s = sum(phi(l,n*(u[m:]-u[:n-m])/m)) + sum(phi(l,n*(1+u[:m]-u[n-m:])/m))
    return (s,pvalue(s,'Morales',l))

# Pardo: test result is inf if n*(u[r1]-u[r2]) = 0 (u[i+m] != u[i-m] for all i) => Take big values for m
def Pardo(u,n,m):
    s = (sum(2*m/(n*(u[m:2*m+1]-u[0]))) + sum(2*m/(n*(u[n-1]-u[n-2*m-1:n-m]))) + sum(2*m/(n*(u[2*m+1:n-1]-u[1:n-2*m-1]))))/n
    return (s,pvalue(s,'Pardo'))

# Cressie and Read 2: test result is nan if l = 0 o l = -1 (l*(l+1) = 0)
def CressieRead2_0_ndiv(u,n,m):
    t = 0
    for i in range(1,m):
        p = u[math.floor(i*n/m)] - u[math.floor((i-1)*n/m)]
        t = t + p*np.log(m*p)
    s = 2*n*t
    return (s,pvalue(s,'Cressie and Read 2 (0)'))

def CressieRead2_ndiv(u,n,m,l):
    t = 0
    if l==0:
        return CressieRead2_0_ndiv(u,n,m)
    else:
        for i in range(1,m):
            p = u[math.floor(i*n/m)] - u[math.floor((i-1)*n/m)]
            t = t + p*((m*p)**l - 1)
        s = 2*n*t/(l*(l+1))
    return (s,pvalue(s,'Cressie and Read 2',l))

def CressieRead2_0(u,n,m):
    p = np.diff(u[::int(n/m)])
    s = 2*n*sum(p*np.log(m*p))
    return (s,pvalue(s,'Cressie and Read 2 (0)'))

def CressieRead2(u,n,m,l):
    t = 0
    if n % m != 0:
        return CressieRead2_ndiv(u,n,m,l)
    elif l == 0:
        return CressieRead2_0(u,n,m)
    else:
        p = np.diff(u[::int(n/m)])
        s = 2*n*sum(p*((m*p)**l - 1))/(l*(l+1))
    return (s,pvalue(s,'Cressie and Read 2',l))

# First four normalized Legendre Polynomials
def Leg(j,x):
    if j == 0:
        return x
    elif j == 1:
        return x**2 - 1/3
    elif j == 2:
        return x**3 - 3*x/5
    else:
        return x**4 - 6*(x**2)/7 + 3/35

# Neyman
def Neyman(u,n,h):
    ne = 0
    for j in range(h):
        ne = ne + sum(Leg(j, u))**2
    s = ne/n
    return (s,pvalue(s,'Neyman'))
        
# Zhang 1: test result is inf if u[i-1] = 0 or 1-u[i-1] = 0
def ZhangA(u,n):
    i = np.arange(1.0, n+1)
    z = sum(np.log(u)/(n-i+1/2) + (np.log(1-u))/(i-1/2))
    return (-z,pvalue(-z,'Zhang 1')) 
    
# Zhang 2: test result is inf if 1/u[i-1]-1 = 0 (u[i-1] cannot be 0 or 1)
def ZhangC(u,n):
    z = sum((np.log(((1/u)-1)/(((n-1/2)/(np.arange(1.0, n+1)-3/4))-1)))**2)
    return (z,pvalue(z,'Zhang 2'))

# Gets the p-value of a test   
def pvalue(s, name, l=None, mode='Two-tail', factor=1):
    pv = get(name,l)(s*factor)
    if mode == 'Two-tail':
        return 2*min(pv, 1-pv)
    elif mode == 'One-tail (r)':
        return pv
    else:
        return 1 - pv

# List with all tests and arguments    
ldist = [(dpos,None,None),(dneg,None,None),(ks_test,'uniform',None),(kuiper_test,None,None),(CramervonMises,None,None),
         (AndersonDarling,None,None),(Watson,None,None),(cpos,None,None),(cneg,None,None),(c,None,None),(k,None,None),
         (t1,None,None),(t2,None,None),(tt1,None,None),(tt2,None,None),(Greenwood,None,None),(QuesenberryMiller,None,None),
         (CressieRead1,-1/2,None),(CressieRead1,2/3,None),(Moran,None,None),(Cressie1,6,None),(Cressie2,3,None),
         (Vasicek,11,None),(Swartz,None,None),(Morales,500,-1/2),(Morales,500,0),(Morales,500,2/3),(Morales,500,1),
         (Pardo,5000,None),(CressieRead2,5,-1/2),(CressieRead2,5,0),(CressieRead2,5,2/3),(CressieRead2,5,1),(Neyman,4,None),
         (ZhangA,None,None),(ZhangC,None,None)]
    
# Executes the test f on a set of sequences u (where a and b are possible arguments of f)
def stat_pv(u,f,a=None,b=None):
    sp = []
    usize = u[0].size
    for i in range(len(u)):
        if (a != None):
            if (b != None):
                sp.append(f(u[i],usize,a,b))
            elif isinstance(a, str):
                sp.append(f(u[i],a))
            else:
                sp.append(f(u[i],usize,a))
        else:
            sp.append(f(u[i],usize))
    s, p = zip(*sp)
    return (s, p)

# Version 1: runs the uniformity tests on all sequences, one by one (using stat_pv)
def utests_v1(u):
    test_sp = []
    for i in range(len(ldist)):
        test_sp.append(stat_pv(u,ldist[i][0],ldist[i][1],ldist[i][2]))
    s,p = zip(*test_sp)
    s = np.array(s).transpose()
    p = np.array(p).transpose()
    return (s,p)    
    
# Executes all uniformity tests on a sequence u, taking advantage of similarities between tests
def uTests(u,h=4):
    sp = [[] for i in range(36)]
    n = len(u)
    v1 = np.arange(1.0, n+1)
    v2 = np.arange(0.0, n)
    
    # D_n +
    dp = (v1/n - u).max()
    sp[0] = (dp, pvalue(dp,'D_n+'))
    
    # D_n -
    dn = (u - v2/n).max()
    sp[1] = (dn, pvalue(dn,'D_n-'))
    
    # Kolmogorov
    kol = max(dp,dn)
    sp[2] = (kol, pvalue(kol,'Kolmogorov'))
    
    # Kuiper
    kui = dp + dn
    sp[3] = (kui, pvalue(kui,'Kuiper'))
    
    # Cramér-von Mises
    cvm = sum((u - (2*v1-1)/(2*n))**2) + 1/(12*n)
    sp[4] = (cvm, pvalue(cvm,'Cramér-von Mises'))
    
    # Anderson–Darling
    ad = -n - (sum((2*v1-1)*(np.log(u) + np.log(1-u[::-1]))))/n
    sp[5] = (ad, pvalue(ad,'Anderson–Darling'))
    
    # Watson
    wat = cvm - n*((sum(u)/n) - 0.5)**2
    sp[6] = (wat, pvalue(wat,'Watson'))
    
    v3 = u-v1/(n+1)
    v4 = u-v2/(n-1)
    
    # C_n+
    cp = v3.max()
    sp[7] = (cp, pvalue(cp,'C_n+'))
    
    # C_n-
    cn = -(v3.min())
    sp[8] = (cn, pvalue(cn,'C_n-'))
    
    # C_n
    c = max(cp,cn)
    sp[9] = (c, pvalue(c,'C_n'))
    
    # K_n
    k = cp + cn
    sp[10] = (k, pvalue(k,'K_n'))
    
    # T_1
    t1 = sum(abs(v3)/n)
    sp[11] = (t1, pvalue(t1,'T_1'))
    
    # T_2
    t2 = sum(((v3)**2)/n)
    sp[12] = (t2, pvalue(t2,'T_2'))
    
    # T_1'
    tt1 = sum(abs(v4)/n)
    sp[13] = (tt1, pvalue(tt1,'TT_1'))
    
    # T_2'
    tt2 = sum(((v4)**2)/n)
    sp[14] = (tt2, pvalue(tt2,'TT_2'))
    
    uu = np.concatenate([[0],u,[1]])
    d = np.diff(uu)

    # Greenwood
    gr = sum(d**2)
    sp[15] = (gr, pvalue(gr,'Greenwood'))
    
    # Quesenberry Miller
    qm = gr + sum(np.multiply(d[1:],d[:n]))
    sp[16] = (qm, pvalue(qm,'Quesenberry Miller'))
    
    # Moran
    mor = -2*sum(np.log((n+1)*d))
    sp[19] = (mor, pvalue(mor,'Moran'))
    
    m = int(n/4)
    uu1 = u[m:] - u[:n-m]
    nuu1 = n*uu1
    uu2 = n*(u[m:2*m+1]-u[0])
    uu3 = n*(u[n-1]-u[n-2*m-1:n-m])
    uu4 = n*(u[2*m+1:n-1]-u[1:n-2*m-1])
    uu5 = np.concatenate([[-u[0]],u,[2-u[n-1]]])
    
    # Cressie 1
    cr1 = np.log(u[m-1]) + sum(np.log(uu1)) + np.log(1 - u[n-m])
    sp[20] = (cr1, pvalue(cr1,'Cressie 1'))
    
    # Cressie 2
    cr2 = (n*u[m-1])**2 + sum(nuu1**2) + (n*(1 - u[n-m]))**2
    sp[21] = (cr2, pvalue(cr2,'Cressie 2'))
    
    # Vasicek
    vas = (sum(np.log(uu2/(2*m))) + sum(np.log(uu3/(2*m))) + sum(np.log(uu4/(2*m))))/n
    sp[22] = (vas, pvalue(vas,'Vasicek'))
    
    # Swartz
    sw = n*(sum(((uu5[2:]-uu5[:n])/2 - 1/n)**2))/2
    sp[23] = (sw, pvalue(sw,'Swartz'))
    
    # Pardo
    par = (sum(2*m/uu2) + sum(2*m/uu3) + sum(2*m/uu4))/n
    sp[28] = (par, pvalue(par,'Pardo'))  
    
    # Neyman
    ne = 0
    for j in range(h):
        ne = ne + sum(Leg(j, u))**2
    ne = ne/n
    sp[33] = (ne, pvalue(ne,'Neyman'))
    
    # Zhang 1
    za = -sum(np.log(u)/(n-v1+1/2) + (np.log(1-u))/(v1-1/2))
    sp[34] = (za, pvalue(za,'Zhang 1'))
    
    # Zhang 2
    zc = sum((np.log(((1/u)-1)/(((n-1/2)/(v1-3/4))-1)))**2)
    sp[35] = (zc, pvalue(zc,'Zhang 2'))
    
    p = np.diff(u[::int(n/m)])
    maux1 = nuu1/m
    maux2 = n*(1+u[:m]-u[n-m:])/m
    l = -1/2
    
    # Cressie and Read 1 (-1/2)
    crr1 = (2*n/(l*(l+1)))*sum(d*((((n+1)*d)**l)-1))
    sp[17] = (crr1, pvalue(crr1,'Cressie and Read 1 (-1/2)'))
    
    # Morales (-1/2)
    mor1 = sum(phi(l,maux1)) + sum(phi(l,maux2))
    sp[24] = (mor1, pvalue(mor1,'Morales (-1/2)'))
    
    # Cressie and Read 2 (-1/2)
    st1 = 2*n*sum(p*((m*p)**l - 1))/(l*(l+1))
    sp[29] = (st1, pvalue(st1,'Cressie and Read 2 (-1/2)'))
        
    l = 0
    
    # Morales (0)
    mor2 = sum(phi(l,maux1)) + sum(phi(l,maux2))
    sp[25] = (mor2, pvalue(mor2,'Morales (0)'))

    # Cressie and Read 2 (0)
    st2 = 2*n*sum(p*np.log(m*p))
    sp[30] = (st2, pvalue(st2,'Cressie and Read 2 (0)'))
    
    l = 2/3
    
    # Cressie and Read 1 (2/3)
    crr2 = (2*n/(l*(l+1)))*sum(d*((((n+1)*d)**l)-1))
    sp[18] = (crr2, pvalue(crr2,'Cressie and Read 1 (2/3)'))
    
    # Morales (2/3)
    mor3 = sum(phi(l,maux1)) + sum(phi(l,maux2))
    sp[26] = (mor3, pvalue(mor3,'Morales (2/3)'))
    
    # Cressie and Read 2 (2/3)
    st3 = 2*n*sum(p*((m*p)**l - 1))/(l*(l+1))
    sp[31] = (st3, pvalue(st3,'Cressie and Read 2 (2/3)'))
    
    l = 1
    
    # Morales (1)
    mor4 = sum(phi(l,maux1)) + sum(phi(l,maux2))
    sp[27] = (mor4, pvalue(mor4,'Morales (1)'))
    
    # Cressie and Read 2 (1)
    st4 = 2*n*sum(p*((m*p)**l - 1))/(l*(l+1))
    sp[32] = (st4, pvalue(st4,'Cressie and Read 2 (1)'))
    
    return sp
    
# Version 2: runs all uniformity tests on each sequence of u (using uTests)
def utests_v2(u):
    l = len(u)
    s = [[] for i in range(l)]
    p = [[] for i in range(l)]
    for i in range(len(u)):
        (s[i], p[i]) = zip(*uTests(u[i]))
    return (s,p)