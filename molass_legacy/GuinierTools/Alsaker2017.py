
"""
    Alsaker2017.py

    converted from file1.R in     
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
from scipy.interpolate import UnivariateSpline
from scipy.linalg import toeplitz
from scipy.sparse import diags
from scipy.linalg import sqrtm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def estimate_Rg(M, num_reps, starting_value=-1):
    if isinstance(starting_value, int):
        starting_value = [starting_value] * num_reps
    
    for val in starting_value:
        if val < -1:
            print("Please enter positive initial values.")
            return
    
    if starting_value[0] != -1:
        choose_sp = True
        print("Initial points input by user.")
        if num_reps > 1:
            for i, val in enumerate(starting_value, start=1):
                if val == 1:
                    print(f"Removed zero points from replicate {i}")
                elif val == 2:
                    print(f"Removed first point from replicate {i}")
                elif val > 2:
                    print(f"Removed first {val-1} points from replicate {i}")
    else:
        choose_sp = False
        starting_value = [1] * num_reps
    
    sp = starting_value
    nreps = num_reps
    M = np.asarray(M)
    
    if nreps == 1:
        # check for zero values in M[:,1]
        first = True
        if M[0, 1] == 0 and first:
            first = False
            i = 0
            while M[i, 1] == 0 and i < len(M[:, 1]) - 1:
                i += 1
            num = i
        print("num=", num)
        if not first:
            # temp_M = M[num+1:, :]
            temp_M = M[num:, :]     # modified
            len_ = len(M[:, 0]) - num
            M = np.zeros((len_, 3))
            M[:, 0] = temp_M[:, 0]
            M[:, 1] = temp_M[:, 1]
            M[:, 2] = temp_M[:, 2]
            n = len_
            out_e = f"Warning: first {num} intensity values equal zero. These values were stripped from the data."
            print(out_e)
        
        first = True
        for i, val in enumerate(M[:, 1]):
            if val < 0 and first:
                n = i
                first = False
                print("Warning: negative intensity values found.")
        if first:
            n = len(M[:, 1])
        
        y3 = np.diff(np.diff(np.diff(np.diff(np.log(M[sp[0]-1:n, 1])))))
        cp2i = cpt_var(y3, n, 7)
        
        if isinstance(cp2i, int) and cp2i == 0:
            cp2i = n - 1
        if cp2i < 60:
            cp2i = 60
        if cp2i > 120:
            cp2i = 120
        
        cp2 = cp2i + sp[0] - 1
        m = 0
        
        comb_spline_fit = comb_spline(M, cp2, cp2i, nreps, sp, m)
        gamma = np.zeros(cp2)
        out = ind_ar_struc(comb_spline_fit, M[sp[0]-1:cp2+sp[0], 1], cp2)
        gamma = out[1:]
        arsum = out[0]
        sigma = create_gamma_matrix(gamma, cp2)
        t = b_v_tradeoff_comb(M, cp2, sigma, comb_spline_fit, nreps, sp, arsum)
        output = calc_Rg(M[:n, :], sigma, t, cp2, nreps, sp, arsum, comb_spline_fit, choose_sp)
        return output
    
    if nreps >= 2:
        n = len(M[:, 1])
        if np.any(M[:, 1] <= 0):
            n = np.where(M[:, 1] <= 0)[0][0]
        for i in range(2, nreps+1):
            if np.any(M[:, i] <= 0):
                temp = np.where(M[:, i] <= 0)[0][0]
                if temp < n:
                    n = temp
        
        cp2i = np.zeros(nreps, dtype=int)
        
        for i in range(nreps):
            y3 = np.diff(np.diff(np.diff(np.diff(np.log(M[sp[i]-1:n, i+1])))))
            cp2i[i] = cpt_var(y3, n, 7)
            if cp2i[i] < 60:
                cp2i[i] = 60
            if cp2i[i] > 120:
                cp2i[i] = 120
        
        cp2 = cp2i + sp - 1
        
        n = min(cp2) - max(sp) + 2
        m = max(cp2) - min(sp) + 1
        
        comb_spline_fit = comb_spline(M, cp2, cp2i, nreps, sp, m)
        gamma = np.zeros((m, nreps))
        arsum = 0
        for i in range(nreps):
            out = ind_ar_struc(comb_spline_fit[(sp[i]-min(sp)):(cp2[i]-min(sp)+1), i], M[sp[i]-1:cp2[i], i+1], m)
            gamma[:, i] = out[1:]
            arsum += out[0]
        arsum /= nreps
        avg_gamma = np.mean(gamma, axis=1)
        
        sigma = create_gamma_matrix(avg_gamma, m)
        t = b_v_tradeoff_comb(M, n, sigma, comb_spline_fit, nreps, sp, arsum)
        cp2 = t + sp - 1
        ep = cp2 + 1
        
        output = calc_Rg(M, sigma, t, n, nreps, sp, arsum, comb_spline_fit, choose_sp)
        
        return output

def comb_spline(M, cp2, cp2i, nreps, sp, m):
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    
    df = 6
    if m > 80:
        df = 8
    
    if nreps > 2:
        ncp = UnivariateSpline(M[min(sp)-1:max(cp2), 0], np.arange(min(sp), max(cp2)+1), k=df)
        
        X = np.ones((cp2[0]-sp[0]+1, 1))
        X = np.hstack((X, np.zeros((cp2[0]-sp[0]+1, nreps-1))))
        X = np.hstack((X, ncp(M[sp[0]-1:cp2[0], 0])))
        Y = np.log(M[sp[0]-1:cp2[0], 1])
        for i in range(1, nreps-1):
            Y = np.concatenate((Y, np.log(M[sp[i]-1:cp2[i], i+1])))
            temp = np.zeros((cp2[i]-sp[i]+1, i))
            temp = np.hstack((temp, np.ones((cp2[i]-sp[i]+1, 1))))
            temp = np.hstack((temp, np.zeros((cp2[i]-sp[i]+1, nreps-i-1))))
            temp = np.hstack((temp, ncp(M[sp[i]-1:cp2[i], 0])))
            X = np.vstack((X, temp))
        Y = np.concatenate((Y, np.log(M[sp[nreps-1]-1:cp2[nreps-1], nreps])))
        temp = np.zeros((cp2[nreps-1]-sp[nreps-1]+1, nreps-1))
        temp = np.hstack((temp, np.ones((cp2[nreps-1]-sp[nreps-1]+1, 1))))
        temp = np.hstack((temp, ncp(M[sp[nreps-1]-1:cp2[nreps-1], 0])))
        X = np.vstack((X, temp))
        beta_est = np.linalg.lstsq(X, Y, rcond=None)[0]
        X = np.ones(m)
        for j in range(1, nreps):
            X = np.hstack((X, np.zeros(m)))
        X = np.hstack((X, ncp(np.linspace(M[min(sp)-1, 0], M[max(cp2)-1, 0], m))))
        beta_curve = X @ beta_est
        
        for i in range(1, nreps):
            X = np.zeros(m)
            for j in range(1, nreps):
                if j == i:
                    X = np.hstack((X, np.ones(m)))
                else:
                    X = np.hstack((X, np.zeros(m)))
            X = np.hstack((X, ncp(np.linspace(M[min(sp)-1, 0], M[max(cp2)-1, 0], m))))
            beta_curve_temp = X @ beta_est
            beta_curve = np.vstack((beta_curve, beta_curve_temp))
        return beta_curve.T
    
    if nreps == 2:
        ncp = UnivariateSpline(M[min(sp)-1:max(cp2), 0], np.arange(min(sp), max(cp2)+1), k=df)
        
        X = np.ones((cp2[0]-sp[0]+1, 1))
        X = np.hstack((X, np.zeros((cp2[0]-sp[0]+1, nreps-1))))
        X = np.hstack((X, ncp(M[sp[0]-1:cp2[0], 0])))
        Y = np.log(M[sp[0]-1:cp2[0], 1])
        
        Y = np.concatenate((Y, np.log(M[sp[1]-1:cp2[1], 2])))
        temp = np.zeros((cp2[1]-sp[1]+1, nreps-1))
        temp = np.hstack((temp, np.ones((cp2[1]-sp[1]+1, 1))))
        temp = np.hstack((temp, ncp(M[sp[1]-1:cp2[1], 0])))
        X = np.vstack((X, temp))
        
        beta_est = np.linalg.lstsq(X, Y, rcond=None)[0]
        X = np.ones(m)
        X = np.hstack((X, np.zeros(m)))
        X = np.hstack((X, ncp(np.linspace(M[min(sp)-1, 0], M[max(cp2)-1, 0], m))))
        beta_curve = X @ beta_est
        
        X = np.zeros(m)
        X = np.hstack((X, np.ones(m)))
        X = np.hstack((X, ncp(np.linspace(M[min(sp)-1, 0], M[max(cp2)-1, 0], m))))
        beta_curve_temp = X @ beta_est
        
        beta_curve = np.vstack((beta_curve, beta_curve_temp))
        
        return beta_curve.T
    
    if nreps == 1:
        X = np.ones((cp2-sp[0]+1, 1))
        # X = np.hstack((X, UnivariateSpline(M[sp[0]-1:cp2+sp[0], 0], np.arange(sp[0], cp2+sp[0]+1), k=df)(M[sp[0]-1:cp2+sp[0], 0])))
        X = np.hstack((X, UnivariateSpline(M[sp[0]-1:cp2+sp[0], 0], np.arange(sp[0], cp2+sp[0]+1))(M[sp[0]-1:cp2+sp[0], 0])))
        Y = np.log(M[sp[0]-1:cp2+sp[0], 1])
        
        beta_est = np.linalg.lstsq(X, Y, rcond=None)[0]
        beta_curve = X @ beta_est
        
        return beta_curve

def ind_ar_struc(comb_spline_fit, d2, m):
    import numpy as np
    from statsmodels.tsa.ar_model import AutoReg
    
    resid = np.log(d2) - comb_spline_fit
    ar_0 = AutoReg(resid, lags=5, old_names=False).fit()
    if ar_0.k_ar == 0:
        phi_temp = 0
    else:
        phi_temp = ar_0.ar_params
    a_0 = {'phi': phi_temp, 'theta': np.array([0]), 'sigma2': ar_0.sigma2}
    arsum = ar_0.sigma2 / (1 - np.sum(ar_0.ar_params))**2
    
    g_0 = np.concatenate(([arsum], ar_0.acf(len=m)[1:]))
    return g_0

def create_gamma_matrix(g, p):
    import numpy as np
    
    gamma = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            gamma[i, j] = g[abs(i-j)]
    return gamma

def cpt_var(y3, n, pen_value):
    import ruptures as rpt
    import molass_legacy.KekLib.DebugPlot as dpl
    
    with dpl.Dp():
        fig, ax = dpl.subplots()
        ax.set_title("cpt_var")
        ax.plot(y3)
        fig.tight_layout()
        dpl.show()

    # cost = numpy_norm_outer(y3)
    # n_cp = pelt(cost, jump=pen_value*np.log(n), min_size=60, n_samples=len(y3))
    # return n_cp.sum()
    # my_bkps = rpt.Pelt(model="l1", jump=int(pen_value*np.log(n)), min_size=60).fit(y3).predict(pen_value*np.log(n))
    # my_bkps = rpt.Pelt(model="l1", jump=5, min_size=3).fit(y3).predict(pen_value*np.log(n))
    my_bkps = rpt.Dynp(model="normal", params={"add_small_diag": False}).fit(y3).predict(pen_value)
    print("my_bkps=", my_bkps, "len(my_bkps)=", len(my_bkps))
    return my_bkps[0]

def b_v_tradeoff_comb(M, cp2, sigma, comb_spline_fit, nreps, sp, arsum):
    #initialize variables
    f = np.full(cp2, 10000000)
    var_est = np.full(cp2, 10000000)
    sum_bias2_avg = np.full(cp2, 10000000)

    s = M[:,0]
    delta_s = np.mean(np.diff(s[:19]))

    #loop though data up to cp2
    for k in range(2, cp2):
        if nreps > 2:
            tempk = k
            k = max(sp) + k - 1
            len_ = np.repeat(k, nreps) - np.repeat(min(sp), nreps) + 2
            lenb = len_ - (sp-min(sp)+1) + 1

            #fit quadratic curve over data
            X = np.column_stack((np.repeat(1, lenb[0]), np.zeros((lenb[0], nreps-1)), M[sp[0]:(k+1),0]**2, M[sp[0]:(k+1),0]**4))
            Y = np.log(M[sp[0]:(k+1),1])

            for i in range(1, nreps-1):
                Y = np.concatenate((Y, np.log(M[sp[i]:(k+1),i+1])))
                temp = np.column_stack((np.zeros((lenb[i], i-1)), np.repeat(1, lenb[i]), np.zeros((lenb[i], nreps-i)), M[sp[i]:(k+1),0]**2, M[sp[i]:(k+1),0]**4))
                X = np.vstack((X, temp))

            Y = np.concatenate((Y, np.log(M[sp[nreps]:(k+1),nreps+1])))
            temp = np.column_stack((np.zeros((lenb[nreps], nreps-1)), np.repeat(1, lenb[nreps]), M[sp[nreps]:(k+1),0]**2, M[sp[nreps]:(k+1),0]**4))
            X = np.vstack((X, temp))

        if nreps == 2:
            tempk = k
            k = max(sp) + k - 1
            len_ = np.repeat(k, nreps) - np.repeat(min(sp), nreps) + 2
            lenb = len_ - (sp-min(sp)+1) + 1

            #fit quadratic curve over data
            X = np.column_stack((np.repeat(1, lenb[0]), np.zeros((lenb[0], nreps-1)), M[sp[0]:(k+1),0]**2, M[sp[0]:(k+1),0]**4))
            Y = np.log(M[sp[0]:(k+1),1])

            Y = np.concatenate((Y, np.log(M[sp[nreps]:(k+1),nreps+1])))
            temp = np.column_stack((np.zeros((lenb[nreps], nreps-1)), np.repeat(1, lenb[nreps]), M[sp[nreps]:(k+1),0]**2, M[sp[nreps]:(k+1),0]**4))
            X = np.vstack((X, temp))

        if nreps == 1:
            tempk = k
            #fit quadratic curve over data
            X = np.column_stack((np.repeat(1, k), M[0:k,0]**2, M[0:k,0]**4))
            Y = np.log(M[0:k,1])

        fit = LinearRegression(fit_intercept=False).fit(X, Y)
        alpha = fit.coef_

        #calculate the estimated bias of Rg
        sum_bias2_avg[tempk] = 9/784*(24*alpha[nreps+2])**2*(k*delta_s)**4

        #calculate variance of Rg
        var_est[tempk] =  (405*arsum)/(nreps*4*k**5*delta_s**4)   

        #bias-variance criterion
        f[tempk] = var_est[tempk] + sum_bias2_avg[tempk]

    #calculate minimum of bias-variance criterion
    t = np.where(f == np.min(f))[0][0]
    return t

def calc_Rg(M, sigma, t, cp2, nreps, sp, arsum, comb_spline_fit, choose_sp):
    s = M[:,0]
    delta_s = np.mean(np.diff(s[:10]))
    output = np.zeros(2)

    if nreps > 2:
        t = max(sp) + t - 1
        len_ = np.repeat(t, nreps) - np.repeat(min(sp), nreps) + 2
        len2 = len_ - (sp-min(sp)+1) + 1
        len3 = len_[0] + min(sp) - 1

        #create design matrix
        X = np.column_stack((np.repeat(1, len2[0]), np.zeros((len2[0], nreps-1)), M[sp[0]:(t+1),0]**2))
        Y = np.log(M[sp[0]:(t+1),1])

        for i in range(1, nreps-1):
            Y = np.concatenate((Y, np.log(M[sp[i]:(t+1),i+1])))
            temp = np.column_stack((np.zeros((len2[i], i-1)), np.repeat(1, len2[i]), np.zeros((len2[i], nreps-i)), M[sp[i]:(t+1),0]**2))
            X = np.vstack((X, temp))

        Y = np.concatenate((Y, np.log(M[sp[nreps]:(t+1),nreps+1])))
        temp = np.column_stack((np.zeros((len2[nreps], nreps-1)), np.repeat(1, len2[nreps]), M[sp[nreps]:(t+1),0]**2))
        X = np.vstack((X, temp))

        #Create covariance matrix
        get = np.column_stack((sigma[(sp[0]-min(sp)+1):len_[0],(sp[0]-min(sp)+1):len_[0]], np.zeros((sum(len2[1:]), len2[0]))))
        for i in range(1, nreps-1):
            temp = np.column_stack((np.zeros((sum(len2[1:i]), len2[i])), sigma[(sp[i]-min(sp)+1):len_[i],(sp[i]-min(sp)+1):len_[i]], np.zeros((sum(len2[(i+1):nreps]), len2[i]))))
            get = np.vstack((get, temp))
        temp = np.column_stack((np.zeros((sum(len2[1:(nreps-1)])), len2[nreps])), sigma[(sp[nreps]-min(sp)+1):len_[nreps],(sp[nreps]-min(sp)+1):len_[nreps]])
        get = np.vstack((get, temp))

        gamma_est = get

    if nreps == 2:
        t = max(sp) + t - 1
        len_ = np.repeat(t, nreps) - np.repeat(min(sp), nreps) + 2
        len2 = len_ - (sp-min(sp)+1) + 1
        len3 = len_[0] + min(sp) - 1

        #create design matrix
        X = np.column_stack((np.repeat(1, len2[0]), np.zeros((len2[0], nreps-1)), M[sp[0]:(t+1),0]**2))
        Y = np.log(M[sp[0]:(t+1),1])

        Y = np.concatenate((Y, np.log(M[sp[nreps]:(t+1),nreps+1])))
        temp = np.column_stack((np.zeros((len2[nreps], nreps-1)), np.repeat(1, len2[nreps]), M[sp[nreps]:(t+1),0]**2))
        X = np.vstack((X, temp))

        #create covariance matrix
        get = np.column_stack((sigma[(sp[0]-min(sp)+1):len_[0],(sp[0]-min(sp)+1):len_[0]], np.zeros((sum(len2[1:]), len2[0]))))
        temp = np.column_stack((np.zeros((sum(len2[1]), len2[nreps])), sigma[(sp[nreps]-min(sp)+1):len_[nreps],(sp[nreps]-min(sp)+1):len_[nreps]]))
        get = np.vstack((get, temp))

        gamma_est = get

    if nreps == 1:
        len3 = t
        X = np.column_stack((np.repeat(1, t), M[sp[0]:(t+sp[0]-1),0]**2))
        Y = np.log(M[sp[0]:(sp[0]+t-1),1])
        gamma_est = sigma[:t,:t]

    fit = LinearRegression(fit_intercept=False).fit(X, Y)
    alpha_curve = fit.predict(X)
    alpha_curve2 = fit.coef_[0] + fit.coef_[nreps]*M[min(sp):len3,0]**2

    e, V = np.linalg.eig(gamma_est)
    B = V @ sqrtm(np.diag(e)) @ V.T
    Y = B @ Y
    X = B @ X
    fit = LinearRegression(fit_intercept=False).fit(X, Y)

    if nreps == 1:
        resids = np.log(M[:cp2,1]) - fit.coef_[0] - fit.coef_[1]*M[:cp2,0]**2
    if nreps > 1:
        resids = np.zeros((cp2, nreps))
        for i in range(nreps):
            resids[:,i] = np.log(M[:cp2,i+1]) - fit.coef_[i] - fit.coef_[nreps]*M[:cp2,0]**2

    #estimate Rg using standard regression technique
    alpha = fit.coef_
    #check for negative Rg
    if alpha[nreps] > 0:
        print("Negative Rg value found. Program stopped.")

    Rg_prev = (-3*alpha[nreps])

    #############################
    #Outlier Detection
    #############################
    if choose_sp == False:
        NUM = t//2
        DFBETAS = np.zeros(NUM)
        d_i = np.zeros(NUM)
        s_d_i = np.zeros(NUM)
        MSE = np.sum((Y-fit.predict(X))**2)/(t-3)
        Y_hat = fit.predict(X)
        d2 = np.log(M[:,1])
        d1 = M[:,0]
        bool_ = True
        i = 0

        while bool_ == True and i < NUM:
            d2_i = d2[1:i+1]
            d1_i = d1[1:i+1]

            M_i = M[i+1:,:]
            comb_spline_fit_i = comb_spline_fit[i+1:]

            t_i = b_v_tradeoff_comb(M_i, cp2-i, sigma, comb_spline_fit_i, nreps, sp, arsum)

            Y_i = np.log(d2_i[:t_i])
            X_i = np.column_stack((np.repeat(1, t_i), (d1_i[:t_i])**2))

            if nreps > 2:
                len_ = np.repeat(t_i, nreps) - np.repeat(1, nreps) + 2
                len2 = len_

                #create design matrix
                X_i = np.column_stack((np.repeat(1, len2[0]), np.zeros((len2[0], nreps-1)), M_i[:t_i+1,0]**2))
                Y_i = np.log(M_i[:t_i+1,1])

                for i in range(1, nreps-1):
                    Y_i = np.concatenate((Y_i, np.log(M[:t_i+1,i+1])))
                    temp = np.column_stack((np.zeros((len2[i], i-1)), np.repeat(1, len2[i]), np.zeros((len2[i], nreps-i)), M_i[:t_i+1,0]**2))
                    X_i = np.vstack((X_i, temp))

                Y_i = np.concatenate((Y_i, np.log(M_i[:t_i+1,nreps+1])))
                temp = np.column_stack((np.zeros((len2[nreps], nreps-1)), np.repeat(1, len2[nreps]), M_i[:t_i+1,0]**2))
                X_i = np.vstack((X_i, temp))

            fit_i = LinearRegression(fit_intercept=False).fit(X_i, Y_i)
            beta_i = fit_i.coef_[nreps]

            Rg_now = (-3*beta_i)
            var_Rg2 = np.sqrt((405*arsum)/(nreps*4*t_i**5*delta_s**4)/9)

            DFBETAS[i] = np.abs(Rg_prev - Rg_now)/np.sqrt(np.var(Y_i))
            Rg_prev = Rg_now
            if DFBETAS[i] < 2/np.sqrt(t_i):
                bool_ = False
            if DFBETAS[i] >= 2/np.sqrt(t_i):
                i += 1

        if i > 0:
            print("Initial points selected automatically.")
            if nreps == 1:
                if i == 1:
                    print("Removed first point from the curve.")
                else:
                    print(f"Removed first {i-1} points from the curve.")
            if nreps > 1:
                if i == 1:
                    print("Removed first point from each curve.")
                else:
                    print(f"Removed first {i-1} points from each curve.")
            sp = np.repeat(i, nreps)
            s = M[:,0]
            delta_s = np.mean(np.diff(s[:10]))
            output = np.zeros(2)

            if nreps > 2:
                t = max(sp) + t - 1
                len_ = np.repeat(t, nreps) - np.repeat(min(sp), nreps) + 2
                len2 = len_ - (sp-min(sp)+1) + 1
                len3 = len_[0] + min(sp) - 1

                #create design matrix
                X = np.column_stack((np.repeat(1, len2[0]), np.zeros((len2[0], nreps-1)), M[sp[0]:(t+1),0]**2))
                Y = np.log(M[sp[0]:(t+1),1])

                for i in range(1, nreps-1):
                    Y = np.concatenate((Y, np.log(M[sp[i]:(t+1),i+1])))
                    temp = np.column_stack((np.zeros((len2[i], i-1)), np.repeat(1, len2[i]), np.zeros((len2[i], nreps-i)), M[sp[i]:(t+1),0]**2))
                    X = np.vstack((X, temp))

                Y = np.concatenate((Y, np.log(M[sp[nreps]:(t+1),nreps+1])))
                temp = np.column_stack((np.zeros((len2[nreps], nreps-1)), np.repeat(1, len2[nreps]), M[sp[nreps]:(t+1),0]**2))
                X = np.vstack((X, temp))

                #Create covariance matrix
                get = np.column_stack((sigma[(sp[0]-min(sp)+1):len_[0],(sp[0]-min(sp)+1):len_[0]], np.zeros((sum(len2[1:]), len2[0]))))
                for i in range(1, nreps-1):
                    temp = np.column_stack((np.zeros((sum(len2[1:i]), len2[i])), sigma[(sp[i]-min(sp)+1):len_[i],(sp[i]-min(sp)+1):len_[i]], np.zeros((sum(len2[(i+1):nreps]), len2[i]))))
                    get = np.vstack((get, temp))
                temp = np.column_stack((np.zeros((sum(len2[1:(nreps-1)])), len2[nreps])), sigma[(sp[nreps]-min(sp)+1):len_[nreps],(sp[nreps]-min(sp)+1):len_[nreps]])
                get = np.vstack((get, temp))

                gamma_est = get

            if nreps == 2:
                t = max(sp) + t - 1
                len_ = np.repeat(t, nreps) - np.repeat(min(sp), nreps) + 2
                len2 = len_ - (sp-min(sp)+1) + 1
                len3 = len_[0] + min(sp) - 1

                #create design matrix
                X = np.column_stack((np.repeat(1, len2[0]), np.zeros((len2[0], nreps-1)), M[sp[0]:(t+1),0]**2))
                Y = np.log(M[sp[0]:(t+1),1])

                Y = np.concatenate((Y, np.log(M[sp[nreps]:(t+1),nreps+1])))
                temp = np.column_stack((np.zeros((len2[nreps], nreps-1)), np.repeat(1, len2[nreps]), M[sp[nreps]:(t+1),0]**2))
                X = np.vstack((X, temp))

                #create covariance matrix
                get = np.column_stack((sigma[(sp[0]-min(sp)+1):len_[0],(sp[0]-min(sp)+1):len_[0]], np.zeros((sum(len2[1:]), len2[0]))))
                temp = np.column_stack((np.zeros((sum(len2[1]), len2[nreps])), sigma[(sp[nreps]-min(sp)+1):len_[nreps],(sp[nreps]-min(sp)+1):len_[nreps]]))
                get = np.vstack((get, temp))

                gamma_est = get

            if nreps == 1:
                len3 = t
                X = np.column_stack((np.repeat(1, t), M[sp[0]:(t+sp[0]-1),0]**2))
                Y = np.log(M[sp[0]:(sp[0]+t-1),1])
                gamma_est = sigma[:t,:t]

            fit = LinearRegression(fit_intercept=False).fit(X, Y)
            alpha_curve = fit.predict(X)
            alpha_curve2 = fit.coef_[0] + fit.coef_[nreps]*M[min(sp):len3,0]**2

            e, V = np.linalg.eig(gamma_est)
            B = V @ sqrtm(np.diag(e)) @ V.T
            Y = B @ Y
            X = B @ X
            fit = LinearRegression(fit_intercept=False).fit(X, Y)

            if nreps == 1:
                resids = np.log(M[:cp2,1]) - fit.coef_[0] - fit.coef_[1]*M[:cp2,0]**2
            if nreps > 1:
                resids = np.zeros((cp2, nreps))
                for i in range(nreps):
                    resids[:,i] = np.log(M[:cp2,i+1]) - fit.coef_[i] - fit.coef_[nreps]*M[:cp2,0]**2

            #estimate Rg using standard regression technique
            alpha = fit.coef_
            #check for negative Rg
            if alpha[nreps] > 0:
                print("Negative Rg value found. Program stopped.")

    ############################

    output[0] = np.sqrt(-3*alpha[nreps])

    #approximate the variance of Rg hat using Taylor linearization
    var_Rg2 = (405*arsum)/(nreps*4*t**5*delta_s**4)/9   
    output[1] = np.sqrt(-3/(4*alpha[nreps])*var_Rg2)

    if nreps == 1:
        #construct plots
        d1 = M[:,0]
        d2 = M[:,1]
        n = len(d1)

        plt.plot(d1[:cp2], resids, 'o', color='blue')
        plt.axhline(0, color='black')
        plt.plot(d1[(t+sp[0]):cp2], resids[(t+sp[0]):cp2], 'o', color='blue')
        plt.plot(d1[sp[0]:(sp[0]+t-1)], resids[sp[0]:(sp[0]+t-1)], 'o', color='red')
        if sp[0] > 1:
            plt.plot(d1[1:(sp[0]-1)], resids[1:(sp[0]-1)], 'o', color='blue')
        plt.legend(["Data points used to fit curve", "Excluded data points"])
        plt.xlabel("S")
        plt.ylabel("Residuals")
        plt.show()

        plt.plot(d1[(sp[0]+t):(sp[0]+cp2-1)]**2, np.log(d2[(sp[0]+t):(sp[0]+cp2-1)]), 'o', color='blue')
        plt.plot(d1[sp[0]:(sp[0]+t-1)]**2, np.log(d2[sp[0]:(sp[0]+t-1)]), 'o', color='red')
        plt.plot(d1[sp[0]:(sp[0]+t-1)]**2, alpha_curve, linewidth=2)
        plt.legend(["Data points used to fit curve", "Fitted curve"])
        plt.xlabel("S^2")
        plt.ylabel("Log(Intensity)")
        plt.show()

        plt.plot(d1[sp[0]:(sp[0]+cp2-1)], np.log(d2[sp[0]:(sp[0]+cp2-1)]), 'o', color='blue')
        plt.plot(d1[sp[0]:(sp[0]+t-1)], np.log(d2[sp[0]:(sp[0]+t-1)]), 'o', color='red')
        plt.plot(d1[sp[0]:(sp[0]+t-1)], alpha_curve, linewidth=2)
        plt.legend(["Data points used to fit curve", "Fitted curve"])
        plt.xlabel("S")
        plt.ylabel("Log(Intensity)")
        plt.show()

    if nreps > 1:
        #construct plots
        d1 = M[:,0]
        d2 = M[:,1]

        plt.plot(d1[:cp2], resids[:,0], 'o', color='blue')
        plt.axhline(0, color='black')
        for i in range(1, nreps):
            d1 = M[:,0]
            d2 = M[:,i+1]-fit.coef_[i]+fit.coef_[0]

            plt.plot(d1[(len3+1):cp2], resids[(len3+1):cp2,i], 'o', color='blue')
            plt.plot(d1[sp[i]:(len3)], resids[sp[i]:(len3),i], 'o', color='red')
            if sp[i] > 1:
                plt.plot(d1[1:(sp[i]-1)], resids[1:(sp[i]-1),i], 'o', color='blue')
        plt.legend(["Data points used to fit curve", "Excluded data points"])
        plt.xlabel("S")
        plt.ylabel("Residuals")
        plt.show()

        plt.plot(d1[(len3+1):(sp[0]+cp2-1)]**2, np.log(d2[(len3+1):(sp[0]+cp2-1)]), 'o', color='blue')
        plt.plot(d1[sp[0]:(len3)]**2, np.log(d2[sp[0]:(len3)]), 'o', color='red')
        plt.plot(d1[min(sp):(len3)]**2, alpha_curve2, linewidth=2)
        plt.legend(["Data points used to fit curve", "Fitted curve"])
        plt.xlabel("S^2")
        plt.ylabel("Log(Intensity)")
        plt.show()

        for i in range(1, nreps):
            d1 = M[:,0]
            d2 = M[:,i+1]-fit.coef_[i]+fit.coef_[0]

            plt.plot(d1[(len3+1):(sp[i]+cp2-1)]**2, np.log(d2[(len3+1):(sp[i]+cp2-1)]), 'o', color='blue')
            plt.plot(d1[sp[i]:(len3)]**2, np.log(d2[sp[i]:(len3)]), 'o', color='red')
            plt.plot(d1[min(sp):(len3)]**2, alpha_curve2, linewidth=2)
            plt.legend(["Data points used to fit curve", "Fitted curve"])
            plt.xlabel("S^2")
            plt.ylabel("Log(Intensity)")
            plt.show()

            plt.plot(d1[(len3+1):(sp[i]+cp2-1)], np.log(d2[(len3+1):(sp[i]+cp2-1)]), 'o', color='blue')
            plt.plot(d1[sp[i]:(len3)], np.log(d2[sp[i]:(len3)]), 'o', color='red')
            plt.plot(d1[min(sp):(len3)], alpha_curve2, linewidth=2)
            plt.legend(["Data points used to fit curve", "Fitted curve"])
            plt.xlabel("S")
            plt.ylabel("Log(Intensity)")
            plt.show()

    #return Rg and its standard deviation
    return np.array([output[0], output[1], t, cp2])