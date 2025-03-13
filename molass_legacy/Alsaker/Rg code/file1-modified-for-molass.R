library(splines);library(MASS);library(changepoint)

estimate_Rg = function(M, num_reps, starting_value = -1, return_y3 = FALSE)
{
  for(i in 1:length(starting_value)){
    if(starting_value[i] < -1){
      print("Please enter positive initial values.")
      return()
    }
  }
  if(starting_value[1] != -1){
    choose_sp = TRUE
    print(paste("Initial points input by user."))
    if(num_reps > 1){
      for(i in 1:num_reps){
        if(starting_value[i] == 1){
          print(paste("Removed zero points from replicate", i))
        }
        if(starting_value[i] == 2){
          print(paste("Removed first point from replicate", i))
        }
        if(starting_value[i] > 2){
          print(paste("Removed first", (starting_value[i]-1), "points from replicate", i))
        }
      }
    }
    if(num_reps == 1){
      if(starting_value == 1){
        print(paste("Removed zero points from the curve."))
      }
      if(starting_value == 2){
        print(paste("Removed first point from the curve."))
      }
      if(starting_value > 2){
        print(paste("Removed first", (starting_value-1), "points from the curve."))
      }    
    }
  }

  if(starting_value[1] == -1){
    choose_sp = FALSE
    starting_value = rep(1,num_reps)
  }

  sp = starting_value
  nreps = num_reps
  if(nreps==1){

    #check for zero values in M[,2]
    first = TRUE
    if(M[1,2] == 0 && first == TRUE)
    {
      first = FALSE
      i = 1
      while(M[i+1,2] == 0 && i < length(M[,2]))
      {
        i = i + 1
      }
      num = i
    }
    if(first == FALSE)
    {
      temp_M = M[(num+1):length(M[,1]),]
      len = length(M[,1]) - num
      M = matrix(rep(0,(len)*3),ncol = 3)
      M[1:len,1] = temp_M[,1]
      M[1:len,2] = temp_M[,2]
      M[1:len,3] = temp_M[,3]
      n = len
      out_e = paste("Warning: first ", num, " intensity values equal zero. These values were stripped from the data.",sep="")
      print(out_e)
    }

    #check for negative values in M[,2]
    first = TRUE
    for(i in 1:length(M[,2]))
    {
      if(M[i,2] < 0 && first == TRUE)
      {
        n = i - 1
        first = FALSE
        print("Warning: negative intensity values found.")
      }
    }
    if(first == TRUE)
    {
      n = length(M[,2])
    }

    y3 = diff(diff(diff(diff(log(M[sp:n,2])))))
    if (return_y3 == TRUE) { return (y3) }
    
    v = cpt.var(y3,know.mean=TRUE,mu=0,test.stat="Normal",method="PELT",penalty="Manual",pen.value=7*log(n))
    cp2i = cpts(v)[1]

    if(is.integer(cpts(v)) && length(cpts(v)) == 0L){cp2i=n-1}
    if(cp2i<60){cp2i=60}
    if(cp2i>120){cp2i=120}

    cp2 = cp2i + sp - 1
    m = 0

    comb_spline_fit = comb_spline(M,cp2,cp2i,nreps,sp,m)
    gamma = rep(0,cp2)
    out = ind_ar_struc(comb_spline_fit,M[sp:(cp2+sp-1),2],cp2)
    gamma = out[-1]
    arsum = out[1]
    sigma = create_gamma_matrix(gamma,cp2)
    t = b_v_tradeoff_comb(M,cp2,sigma,comb_spline_fit,nreps,sp,arsum)
    output = calc_Rg(M[1:n,],sigma,t,cp2,nreps,sp,arsum,comb_spline_fit,choose_sp)
    return(output)
  }

  if(nreps>=2){
    n = length(M[,2])
    if(sum(M[,2]<=0)>0){
      n = min(which(M[,2]<=0)) - 1
    }
    for(i in 2:nreps){
      if(sum(M[,i+1]<=0)>0){
        temp = min(which(M[,i+1]<=0)) - 1
        if(temp < n){
          n = temp
        }
      }
    }

    cp2i = rep(0,nreps)

    for(i in 1:nreps){
      y3 = diff(diff(diff(diff(log(M[sp[i]:n,i+1])))))
      v = cpt.var(y3,know.mean=TRUE,mu=0,test.stat="Normal",method="PELT",penalty="Manual",pen.value=7*log(n))
      cp2i[i] = cpts(v)[1]
      if(cp2i[i]<60){cp2i[i]=60}
      if(cp2i[i]>120){cp2i[i]=120}
    }
    cp2 = cp2i + sp - 1  

    n = min(cp2) - max(sp) + 1
    m = max(cp2) - min(sp) + 1

    comb_spline_fit = comb_spline(M,cp2,cp2i,nreps,sp,m)
    gamma = array(0,dim=c(m,nreps))
    arsum = 0
    for(i in 1:nreps){
      out  = ind_ar_struc(comb_spline_fit[(sp[i]-min(sp)+1):(cp2[i]-min(sp)+1),i],M[sp[i]:cp2[i],i+1],m)
      gamma[,i] = out[-1]
      arsum[i] = out[1]
   }
    arsum = mean(arsum)
    avg_gamma = rep(0,m)
    for(i in 1:m){
      avg_gamma[i] = mean(gamma[i,])
    }

    sigma = create_gamma_matrix(avg_gamma,m)
    t = b_v_tradeoff_comb(M,n,sigma,comb_spline_fit,nreps,sp,arsum)
    cp2 = rep(t,nreps) + sp - 1 
    ep = cp2 + 1

    output = calc_Rg(M,sigma,t,n,nreps,sp,arsum,comb_spline_fit,choose_sp)
    return(output)
  }
}

comb_spline = function(M,cp2,cp2i,nreps,sp,m)
{
  #set number of knots for cubic spline fit of data 
  df = 6
  if(m>80){ df = 8 }

  #fit spine curve to data
  if(nreps>2){
    ncp = ns(M[min(sp):max(cp2),1],df)

    X = cbind(rep(1,cp2[1]-sp[1]+1),matrix(0,nrow=(cp2[1]-sp[1]+1),ncol=(nreps-1)),ncp[(sp[1]-min(sp)+1):(cp2[1]-min(sp)+1),])
    Y = log(M[sp[1]:cp2[1],2])
    for(i in 2:(nreps-1)){
      Y = c(Y,log(M[sp[i]:cp2[i],i+1]))
      temp = cbind(matrix(0,nrow=(cp2[i]-sp[i]+1),ncol=(i-1)),rep(1,cp2[i]-sp[i]+1),matrix(0,nrow=(cp2[i]-sp[i]+1),ncol=(nreps-i)),ncp[(sp[i]-min(sp)+1):(cp2[i]-min(sp)+1),])

      X = rbind(X, temp)
    }
    Y = c(Y,log(M[sp[nreps]:cp2[nreps],nreps+1]))
    temp = cbind(matrix(0,nrow=(cp2[nreps]-sp[nreps]+1),ncol=(nreps-1)),rep(1,cp2[nreps]-sp[nreps]+1),ncp[(sp[nreps]-min(sp)+1):(cp2[nreps]-min(sp)+1),])
    X = rbind(X, temp)
    fit = lm(Y ~ X-1)
    beta_est = fit$coefficients
    X = rep(1,m)
    for(j in 2:nreps){
      X = cbind(X, rep(0,m))
    }
    X = cbind(X,ncp)
    beta_curve = X %*% beta_est

    #create fitted curve
    for(i in 2:nreps){  
      X = rep(0,m)
      for(j in 2:nreps){
        if(j == i){
          X = cbind(X, rep(1,m))
        }
        if(j != i){
          X = cbind(X, rep(0,m))
        }
      }
      X = cbind(X,ncp)
      beta_curve_temp = X %*% beta_est
      beta_curve = cbind(beta_curve, beta_curve_temp)
    }  
    return(beta_curve)
  }

  if(nreps==2){
    ncp = ns(M[min(sp):max(cp2),1],df)

    X = cbind(rep(1,cp2[1]-sp[1]+1),matrix(0,nrow=(cp2[1]-sp[1]+1),ncol=(nreps-1)),ncp[(sp[1]-min(sp)+1):(cp2[1]-min(sp)+1),])
    Y = log(M[sp[1]:cp2[1],2])

    Y = c(Y,log(M[sp[nreps]:cp2[nreps],nreps+1]))
    temp = cbind(matrix(0,nrow=(cp2[nreps]-sp[nreps]+1),ncol=(nreps-1)),rep(1,cp2[nreps]-sp[nreps]+1),ncp[(sp[nreps]-min(sp)+1):(cp2[nreps]-min(sp)+1),])
    X = rbind(X, temp)

    fit = lm(Y ~ X-1)
    beta_est = fit$coefficients
    X = rep(1,m)
    for(j in 2:nreps){
      X = cbind(X, rep(0,m))
    }
    X = cbind(X,ncp)
    beta_curve = X %*% beta_est

    X = rep(0,m)
    X = cbind(X, rep(1,m))

    X = cbind(X,ncp)
    beta_curve_temp = X %*% beta_est

    beta_curve = cbind(beta_curve, beta_curve_temp)
   
    return(beta_curve)
  }

  if(nreps==1){
    X = cbind(rep(1,cp2),ns(M[sp[1]:(cp2+sp[1]-1),1],df))
    Y = log(M[sp[1]:(cp2+sp[1]-1),2])

    fit = lm(Y ~ X-1)
    beta_est = fit$coefficients
    beta_curve = X %*% beta_est

    return(beta_curve)
  }
}

ind_ar_struc = function(comb_spline_fit,d2,m)
{
  resid = log(d2) - comb_spline_fit
  ar_0 = ar(resid,order.max=5)
  #extract ar data and put into variable a_0
  if(ar_0$order == 0)
  {
    phi_temp = 0
  }
  if(ar_0$order > 0)
  {
    phi_temp = ar_0$ar
  }
  a_0 = list(phi = phi_temp, theta = c(0), sigma2 = ar_0$var.pred)
  arsum = ar_0$var.pred/(1-sum(ar_0$ar))^2

  #create autocovariance vector
  g_0 = unlist(ARMAacf(a_0$phi,lag.max = m)*ar_0$var.pred)
  return(c(arsum,g_0[1:m]))
}

create_gamma_matrix = function(g,p)
{
  x=rep(0,p^2)
  gamma=matrix(x,p)
  for( i in 1:p )
  {
    for( j in 1:p )
    {
      gamma[i,j] = g[abs(i-j)+1]
    }
  }
  return(gamma)
}

b_v_tradeoff_comb = function(M,cp2,sigma,comb_spline_fit,nreps,sp,arsum)
{
  #initialize variables
  f = rep(10000000,cp2)
  var_est = rep(10000000,cp2)
  sum_bias2_avg = rep(10000000,cp2)

  s = M[,1]
  delta.s=mean(diff(s[1:19]))

  #loop though data up to cp2
  for(k in 3:(cp2-1))
  {
    if(nreps>2){
      tempk = k
      k = max(sp) + k - 1
      len = rep(k,nreps) - rep(min(sp),nreps) + 2
      lenb = len - (sp-min(sp)+1) + 1

      #fit quadratic curve over data
      X = cbind(rep(1,lenb[1]),matrix(0,nrow=lenb[1],ncol=(nreps-1)),M[sp[1]:(k+1),1]^2,M[sp[1]:(k+1),1]^4)
      Y = log(M[sp[1]:(k+1),2])

      for(i in 2:(nreps-1)){
        Y = c(Y,log(M[sp[i]:(k+1),i+1]))
        temp = cbind(matrix(0,nrow=lenb[i],ncol=(i-1)),rep(1,lenb[i]),matrix(0,nrow=lenb[i],ncol=(nreps-i)),M[sp[i]:(k+1),1]^2,M[sp[i]:(k+1),1]^4)
        X = rbind(X, temp)
      }

      Y = c(Y,log(M[sp[nreps]:(k+1),nreps+1]))
      temp = cbind(matrix(0,nrow=lenb[nreps],ncol=(nreps-1)),rep(1,lenb[nreps]),M[sp[nreps]:(k+1),1]^2,M[sp[nreps]:(k+1),1]^4)
      X = rbind(X, temp)

    }
    if(nreps==2){
      tempk = k
      k = max(sp) + k - 1
      len = rep(k,nreps) - rep(min(sp),nreps) + 2
      lenb = len - (sp-min(sp)+1) + 1

      #fit quadratic curve over data
      X = cbind(rep(1,lenb[1]),matrix(0,nrow=lenb[1],ncol=(nreps-1)),M[sp[1]:(k+1),1]^2,M[sp[1]:(k+1),1]^4)
      Y = log(M[sp[1]:(k+1),2])

      Y = c(Y,log(M[sp[nreps]:(k+1),nreps+1]))
      temp = cbind(matrix(0,nrow=lenb[nreps],ncol=(nreps-1)),rep(1,lenb[nreps]),M[sp[nreps]:(k+1),1]^2,M[sp[nreps]:(k+1),1]^4)
      X = rbind(X, temp)
    }
    if(nreps==1){
      tempk = k
      #fit quadratic curve over data
      X = cbind(rep(1,k),M[1:k,1]^2,M[1:k,1]^4)
      Y = log(M[1:k,2])
    }  

    fit = lm(Y~X-1)
    alpha = fit$coeff

    #calculate the estimated bias of Rg
    sum_bias2_avg[tempk] = 9/784*(24*alpha[nreps+2])^2*(k*delta.s)^4

    #calculate variance of Rg
    var_est[tempk] =  (405*arsum)/(nreps*4*k^5*delta.s^4)   

    #bias-variance criterion
    f[tempk] = var_est[tempk]+sum_bias2_avg[tempk]
  }

  #calculate minimum of bias-variance criterion
  t = which(f == min(f))
  return(t)
}

calc_Rg = function(M,sigma,t,cp2,nreps,sp,arsum,comb_spline_fit,choose_sp)
{
  s = M[,1]
  delta.s=mean(diff(s[1:10]))
  output = c(0,0)

  if(nreps>2){
    t = max(sp) + t - 1
    len = rep(t,nreps) - rep(min(sp),nreps) + 2
    len2 = len - (sp-min(sp)+1) + 1
    len3 = len[1] + min(sp) - 1

    #create design matrix
    X = cbind(rep(1,len2[1]),matrix(0,nrow=len2[1],ncol=(nreps-1)),M[sp[1]:(t+1),1]^2)
    Y = log(M[sp[1]:(t+1),2])

    for(i in 2:(nreps-1)){
      Y = c(Y,log(M[sp[i]:(t+1),i+1]))
      temp = cbind(matrix(0,nrow=len2[i],ncol=(i-1)),rep(1,len2[i]),matrix(0,nrow=len2[i],ncol=(nreps-i)),M[sp[i]:(t+1),1]^2)
      X = rbind(X, temp)
    }
    Y = c(Y,log(M[sp[nreps]:(t+1),nreps+1]))
    temp = cbind(matrix(0,nrow=len2[nreps],ncol=(nreps-1)),rep(1,len2[nreps]),M[sp[nreps]:(t+1),1]^2)
    X = rbind(X, temp)

    #Create covariance matrix
    get = cbind(sigma[(sp[1]-min(sp)+1):len[1],(sp[1]-min(sp)+1):len[1]],matrix(0,ncol=(sum(len2[-1])),nrow=len2[1]))
    for(i in 2:(nreps-1)){
      temp = cbind(matrix(0,ncol=(sum(len2[1:(i-1)])),nrow=len2[i]),sigma[(sp[i]-min(sp)+1):len[i],(sp[i]-min(sp)+1):len[i]],matrix(0,ncol=(sum(len2[(i+1):nreps])),nrow=len2[i]))
      get = rbind(get, temp)
    }
    temp = cbind(matrix(0,ncol=(sum(len2[1:(nreps-1)])),nrow=len2[nreps]),sigma[(sp[nreps]-min(sp)+1):len[nreps],(sp[nreps]-min(sp)+1):len[nreps]])
    get = rbind(get, temp)

    gamma_est = get
  }

  if(nreps==2){
    t = max(sp) + t - 1
    len = rep(t,nreps) - rep(min(sp),nreps) + 2
    len2 = len - (sp-min(sp)+1) + 1
    len3 = len[1] + min(sp) - 1

    #create design matrix
    X = cbind(rep(1,len2[1]),matrix(0,nrow=len2[1],ncol=(nreps-1)),M[sp[1]:(t+1),1]^2)
    Y = log(M[sp[1]:(t+1),2])

    Y = c(Y,log(M[sp[nreps]:(t+1),nreps+1]))
    temp = cbind(matrix(0,nrow=len2[nreps],ncol=(nreps-1)),rep(1,len2[nreps]),M[sp[nreps]:(t+1),1]^2)
    X = rbind(X, temp)

    #create covariance matrix
    get = cbind(sigma[(sp[1]-min(sp)+1):len[1],(sp[1]-min(sp)+1):len[1]],matrix(0,ncol=(sum(len2[-1])),nrow=len2[1]))
    temp = cbind(matrix(0,ncol=sum(len2[1]),nrow=len2[nreps]),sigma[(sp[nreps]-min(sp)+1):len[nreps],(sp[nreps]-min(sp)+1):len[nreps]])
    get = rbind(get, temp)

    gamma_est = get
  }

  if(nreps==1){
    len3 = t
    X = cbind(rep(1,t),M[sp[1]:(t+sp[1]-1),1]^2)
    Y = log(M[sp[1]:(sp[1]+t-1),2])
    gamma_est = sigma[1:t,1:t]
  }
  fit = lm(Y~X-1)
  alpha_curve = fit$fitted
  alpha_curve2 = fit$coef[1] + fit$coef[nreps+1]*M[min(sp):len3,1]^2


  e = eigen(gamma_est)
  V = e$vectors
  B = V %*% diag(sqrt(e$values)) %*% t(V)
  Y = B%*%Y
  X = B%*%X
  fit = lm(Y~X-1)

  if(nreps == 1){
    resids = log(M[1:cp2,2]) - fit$coef[1] - fit$coef[2]*M[1:cp2,1]^2
  }
  if(nreps > 1){
    resids = matrix(0,ncol=nreps,nrow=cp2)
    for(i in 1:nreps){
      resids[,i] = log(M[1:cp2,1+i]) - fit$coef[i] - fit$coef[nreps+1]*M[1:cp2,1]^2
    }
  }

  #estimate Rg using standard regression technique
  alpha = fit$coeff
  #check for negative Rg
  if( alpha[nreps+1] > 0 )
  {
    print("Negative Rg value found. Program stopped.")
  }

  Rg_prev = (-3*alpha[nreps+1])

#############################
#Outlier Detection
#############################
if(choose_sp == FALSE){

  NUM = t/2
  DFBETAS = rep(0,NUM)
  d_i = rep(0,NUM)
  s_d_i = rep(0,NUM)
  MSE = sum((Y-fit$fitted)^2)/(t-3)
  Y_hat = fit$fitted
  d2 = log(M[,2])
  d1 = M[,1]
  bool = TRUE
  i = 1

  while(bool == TRUE & i < NUM){
    d2_i = d2[-(1:i)]
    d1_i = d1[-(1:i)]

    M_i = M[-(1:i),]
    comb_spline_fit_i = comb_spline_fit[-(1:i)]

    t_i = b_v_tradeoff_comb(M_i,cp2-i,sigma,comb_spline_fit_i,nreps,sp,arsum)

    Y_i = log(d2_i[1:t_i])
    X_i = cbind(rep(1,t_i),(d1_i[1:t_i])^2)

    if(nreps>2){
      len = rep(t_i,nreps) - rep(1,nreps) + 2
      len2 = len 

      #create design matrix
      X_i = cbind(rep(1,len2[1]),matrix(0,nrow=len2[1],ncol=(nreps-1)),M_i[1:(t_i+1),1]^2)
      Y_i = log(M_i[1:(t_i+1),2])

      for(i in 2:(nreps-1)){
        Y_i = c(Y_i,log(M[1:(t_i+1),i+1]))
        temp = cbind(matrix(0,nrow=len2[i],ncol=(i-1)),rep(1,len2[i]),matrix(0,nrow=len2[i],ncol=(nreps-i)),M_i[1:(t_i+1),1]^2)
        X_i = rbind(X_i, temp)
      }
      Y_i = c(Y_i,log(M_i[1:(t_i+1),nreps+1]))
      temp = cbind(matrix(0,nrow=len2[nreps],ncol=(nreps-1)),rep(1,len2[nreps]),M_i[1:(t_i+1),1]^2)
      X_i = rbind(X_i, temp)
    }

    fit_i = lm(Y_i~X_i-1)
    beta_i = summary(fit_i)$coef[nreps+1,1]


    Rg_now = (-3*beta_i)
    var_Rg2 = sqrt((405*arsum)/(nreps*4*t_i^5*delta.s^4)/9)
  

    DFBETAS[i] = abs(Rg_prev - Rg_now)/summary(fit_i)$coef[nreps+1,2]#var_Rg2
    Rg_prev = Rg_now
    if( DFBETAS[i] < 2/sqrt(t_i) ){
      bool = FALSE
    }
    if( DFBETAS[i] >= 2/sqrt(t_i) ){
      i = i + 1
    }
  }

  #print("DEBETAS")
  #print(DFBETAS)
  #print(i)

  if( i > 1 ){
    print(paste("Initial points selected automatically."))
    if(nreps == 1){
      if(i == 2){
        print(paste("Removed first point from the curve."))
      }
      if(i != 2){
        print(paste("Removed first", (i-1), "points from the curve."))
      }
    }
    if(nreps > 1){
      if(i == 2){
        print(paste("Removed first point from each curve."))
      }
      if(i != 2){
        print(paste("Removed first", (i-1), "points from each curve."))
      }    }
    sp = rep(i,nreps)
    s = M[,1]
    delta.s=mean(diff(s[1:10]))
    output = c(0,0)

    if(nreps>2){
      t = max(sp) + t - 1
      len = rep(t,nreps) - rep(min(sp),nreps) + 2
      len2 = len - (sp-min(sp)+1) + 1
      len3 = len[1] + min(sp) - 1

      #create design matrix
      X = cbind(rep(1,len2[1]),matrix(0,nrow=len2[1],ncol=(nreps-1)),M[sp[1]:(t+1),1]^2)
      Y = log(M[sp[1]:(t+1),2])

      for(i in 2:(nreps-1)){
        Y = c(Y,log(M[sp[i]:(t+1),i+1]))
        temp = cbind(matrix(0,nrow=len2[i],ncol=(i-1)),rep(1,len2[i]),matrix(0,nrow=len2[i],ncol=(nreps-i)),M[sp[i]:(t+1),1]^2)
        X = rbind(X, temp)
      }
      Y = c(Y,log(M[sp[nreps]:(t+1),nreps+1]))
      temp = cbind(matrix(0,nrow=len2[nreps],ncol=(nreps-1)),rep(1,len2[nreps]),M[sp[nreps]:(t+1),1]^2)
      X = rbind(X, temp)

      #Create covariance matrix
      get = cbind(sigma[(sp[1]-min(sp)+1):len[1],(sp[1]-min(sp)+1):len[1]],matrix(0,ncol=(sum(len2[-1])),nrow=len2[1]))
      for(i in 2:(nreps-1)){
        temp = cbind(matrix(0,ncol=(sum(len2[1:(i-1)])),nrow=len2[i]),sigma[(sp[i]-min(sp)+1):len[i],(sp[i]-min(sp)+1):len[i]],matrix(0,ncol=(sum(len2[(i+1):nreps])),nrow=len2[i]))
        get = rbind(get, temp)
      }
      temp = cbind(matrix(0,ncol=(sum(len2[1:(nreps-1)])),nrow=len2[nreps]),sigma[(sp[nreps]-min(sp)+1):len[nreps],(sp[nreps]-min(sp)+1):len[nreps]])
      get = rbind(get, temp)

      gamma_est = get
    }

    if(nreps==2){
      t = max(sp) + t - 1
      len = rep(t,nreps) - rep(min(sp),nreps) + 2
      len2 = len - (sp-min(sp)+1) + 1
      len3 = len[1] + min(sp) - 1

      #create design matrix
      X = cbind(rep(1,len2[1]),matrix(0,nrow=len2[1],ncol=(nreps-1)),M[sp[1]:(t+1),1]^2)
      Y = log(M[sp[1]:(t+1),2])

      Y = c(Y,log(M[sp[nreps]:(t+1),nreps+1]))
      temp = cbind(matrix(0,nrow=len2[nreps],ncol=(nreps-1)),rep(1,len2[nreps]),M[sp[nreps]:(t+1),1]^2)
      X = rbind(X, temp)

      #create covariance matrix
      get = cbind(sigma[(sp[1]-min(sp)+1):len[1],(sp[1]-min(sp)+1):len[1]],matrix(0,ncol=(sum(len2[-1])),nrow=len2[1]))
      temp = cbind(matrix(0,ncol=sum(len2[1]),nrow=len2[nreps]),sigma[(sp[nreps]-min(sp)+1):len[nreps],(sp[nreps]-min(sp)+1):len[nreps]])
      get = rbind(get, temp)

      gamma_est = get
    }

    if(nreps==1){
      len3 = t
      X = cbind(rep(1,t),M[sp[1]:(t+sp[1]-1),1]^2)
      Y = log(M[sp[1]:(sp[1]+t-1),2])
      gamma_est = sigma[1:t,1:t]
    }
    fit = lm(Y~X-1)
    alpha_curve = fit$fitted
    alpha_curve2 = fit$coef[1] + fit$coef[nreps+1]*M[min(sp):len3,1]^2

    e = eigen(gamma_est)
    V = e$vectors
    B = V %*% diag(sqrt(e$values)) %*% t(V)
    Y = B%*%Y
    X = B%*%X
    fit = lm(Y~X-1)

    if(nreps == 1){
      resids = log(M[1:cp2,2]) - fit$coef[1] - fit$coef[2]*M[1:cp2,1]^2
    }
    if(nreps > 1){
      resids = matrix(0,ncol=nreps,nrow=cp2)
      for(i in 1:nreps){
        resids[,i] = log(M[1:cp2,1+i]) - fit$coef[i] - fit$coef[nreps+1]*M[1:cp2,1]^2
      }
    }

    #estimate Rg using standard regression technique
    alpha = fit$coeff
    #check for negative Rg
    if( alpha[nreps+1] > 0 )
    {
      print("Negative Rg value found. Program stopped.")
    }
  }
}
############################

  output[1] = sqrt(-3*alpha[nreps+1])

  #approximate the variance of Rg hat using Taylor linearization
  var_Rg2 = (405*arsum)/(nreps*4*t^5*delta.s^4)/9   
  output[2]= sqrt(-3/(4*alpha[nreps+1])*var_Rg2)

  print("------------------- nreps=", nreps)

if(FALSE){
  if(nreps == 1){
    #construct plots
    d1 = M[,1]
    d2 = M[,2]
    n = length(d1)

    plot(d1[1:(cp2)],resids,ylab="Residuals",xlab="S",pch=1,col="blue",cex=0.75,type="n")
    abline(h=0)
    points(d1[((t+sp[1]):cp2)],resids[((t+sp[1]):cp2)],ylab="Residuals",xlab="S",pch=1,col="blue",cex=0.75)
    points(d1[sp[1]:(sp[1]+t-1)],resids[sp[1]:(sp[1]+t-1)],col="red",pch=20,cex=0.75)
    if(sp > 1){
      points(d1[1:(sp[1]-1)],resids[1:(sp[1]-1)],pch=1,col="blue",cex=0.75)  
      #abline(v = (d1[sp[1]]-delta.s/2))
    }
    #abline(v = (d1[sp[1]+t-1]+delta.s/2))
    legend("topright", c("Data points used to fit curve","Excluded data points"),cex=.8,pch=c(20,1), col=c("red","blue"), lty=c(0,0));
 
    windows()

    xspace = (max(d1[1:(1+cp2-1)]^2)-min(d1[1:(1+cp2-1)]^2))/10
    yspace = (max(log(d2[1:(1+cp2-1)]))-min(log(d2[1:(1+cp2-1)])))/100
    plot(d1[(sp[1]+t):(sp[1]+cp2-1)]^2,log(d2[(sp[1]+t):(sp[1]+cp2-1)]),xlim=c(min(d1[1:(sp[1]+cp2-1)]^2),max(d1[sp[1]:(sp[1]+cp2-1)]^2)),ylim=c(min(log(d2[1:(sp[1]+cp2-1)])),max(log(d2[1:(sp[1]+cp2-1)]))),xlab=expression(S^2),ylab="Log(Intensity)",pch=1,col="blue",cex=0.75)
    points(d1[sp[1]:(sp[1]+t-1)]^2,log(d2[sp[1]:(sp[1]+t-1)]),col="red",pch=20,cex=0.75)  
    if(sp > 1){
      points(d1[1:(sp[1]-1)]^2,log(d2[1:(sp[1]-1)]),pch=1,col="blue",cex=0.75)  
    }
    lines(d1[sp[1]:(sp[1]+t-1)]^2,alpha_curve,lwd=2)
    legend("topright", c("Data points used to fit curve","Fitted curve"),cex=.8,pch=c(20,NA), col=c("red","black"), lty=0:1);
    rg = format(round(output[1],1),nsmall=1) ;text((d1[sp[1]]^2+xspace),(log(d2[(sp[1]+cp2-1)])+yspace*15),bquote(hat(R)[g] == .(rg)))
    serg = format(round(output[2],2),nsmall=2) ; text((d1[sp[1]]^2+xspace*1.9),(log(d2[(sp[1]+cp2-1)])+yspace*9.8),paste("Std. Deviation = ",serg,sep=""))

    windows()

    xspace = (max(d1)-min(d1))/10
    yspace = (max(log(d2))-min(log(d2)))/100
    plot(d1[-(1:(sp[1]+t-1))],log(d2[-(1:(sp[1]+t-1))]),xlim=c(min(d1),max(d1)),ylim=c(min(log(d2[1:n])),max(log(d2[1:(sp[1]+cp2-1)]))),xlab="S",ylab="Log(Intensity)",pch=1,col="blue",cex=0.75)
    lines(d1[sp[1]:(sp[1]+t-1)],alpha_curve)
    text((d1[n]-xspace*4),max(log(d2[c(1,n)])-yspace*20),bquote(hat(R)[g] == .(rg)))
    text((d1[n]-xspace*3.1),(max(log(d2[c(1,n)]))-yspace*26),paste("Std. Deviation = ",serg,sep=""))
    points(d1[sp[1]:(sp[1]+t-1)],log(d2[sp[1]:(sp[1]+t-1)]),col="red",pch=20,cex=0.75)
    if(sp > 1){
      points(d1[1:(sp[1]-1)],log(d2[1:(sp[1]-1)]),pch=1,col="blue",cex=0.75)  
    }
    lines(d1[sp[1]:(sp[1]+t-1)],alpha_curve,lwd=1.7)
    legend("topright", c("Data points used to fit curve","Fitted curve","Bias-variance criterion range"),cex=.8,pch=c(20,NA,NA), col=c("red","black","black"), lty=0:2);
    lines(d1[sp[1]:(sp[1]+cp2-1)],rep(max(log(d2))+.2,cp2),lwd=2,lt=2)
  }

  if(nreps > 1){
    #construct plots
    d1 = M[,1]
    d2 = M[,2]

    plot(d1[1:(cp2)],resids[,1],ylab="Residuals",xlab="S",pch=1,col="blue",cex=0.75,type="n")
    abline(h=0)
    for(i in 1:nreps){
      d1 = M[,1]
      d2 = M[,i+1]-fit$coef[i]+fit$coef[1]

      points(d1[(len3+1):cp2],resids[(len3+1):cp2,i],ylab="Residuals",xlab="S",pch=1,col="blue",cex=0.75)
      points(d1[sp[i]:(len3)],resids[sp[i]:(len3),i],col="red",pch=20,cex=0.75)
      if(sp[i] > 1){
        points(d1[1:(sp[i]-1)],resids[1:(sp[i]-1),i],pch=1,col="blue",cex=0.75)  
        #abline(v = (d1[sp[1]]-delta.s/2))
      }
    }
    #abline(v = (d1[sp[1]+t-1]+delta.s/2))
    legend("topright", c("Data points used to fit curve","Excluded data points"),cex=.8,pch=c(20,1), col=c("red","blue"), lty=c(0,0));
  
    windows()

    n = length(d1)
    xspace = (max(d1[1:(1+cp2-1)]^2)-min(d1[1:(1+cp2-1)]^2))/10
    yspace = (max(log(d2[1:(1+cp2-1)]))-min(log(d2[1:(1+cp2-1)])))/100
    plot(d1[(len3+1):(sp[1]+cp2-1)]^2,log(d2[(len3+1):(sp[1]+cp2-1)]),xlim=c(min(d1[1:(1+cp2-1)]^2),max(d1[1:(1+cp2-1)]^2)),ylim=c(min(log(d2[1:(1+cp2-1)])),max(log(d2[1:(1+cp2-1)]))),xlab=expression(S^2),ylab="Log(Intensity)",pch=1,col="blue",cex=0.75)
    points(d1[sp[1]:(len3)]^2,log(d2[sp[1]:(len3)]),col="red",pch=20,cex=1/nreps)    

    legend("topright", c("Data points used to fit curve","Fitted curve"),cex=.8,pch=c(20,NA), col=c("red","black"), lty=0:1);
    rg = format(round(output[1],1),nsmall=1) ;text((d1[1]^2+xspace),(log(d2[(1+cp2-1)])+yspace*15),bquote(hat(R)[g] == .(rg)))
    serg = format(round(output[2],2),nsmall=2) ; text((d1[1]^2+xspace*1.9),(log(d2[(1+cp2-1)])+yspace*9.8),paste("Std. Deviation = ",serg,sep=""))

    for(i in 2:nreps){
      d1 = M[,1]
      d2 = M[,i+1]-fit$coef[i]+fit$coef[1]
      n = length(d1)
      points(d1[(len3+1):(sp[i]+cp2-1)]^2,log(d2[(len3+1):(sp[i]+cp2-1)]),pch=1,col="blue",cex=0.75)###xlim=c(min(d1[sp[1]:(sp[1]+cp2-1)]^2),max(d1[sp[1]:(sp[1]+cp2-1)]^2)),ylim=c(min(log(d2[sp[1]:(sp[1]+cp2-1)])),max(log(d2[sp[1]:(sp[1]+cp2-1)]))),xlab=expression(S^2),ylab="Log(Intensity)")
      points(d1[sp[i]:(len3)]^2,log(d2[sp[i]:(len3)]),col="red",pch=20,cex=1/nreps) 
      if(sp[i] > 1){
        points(d1[1:(sp[i]-1)]^2,log(d2[1:(sp[i]-1)]),pch=1,col="blue",cex=0.75)  
      }
    }

    lines(d1[min(sp):(len3)]^2,alpha_curve2,lwd=2)

    windows()
   
    d1 = M[,1]
    d2 = M[,2]
    xspace = (max(d1)-min(d1))/10
    yspace = (max(log(d2))-min(log(d2)))/100
    plot(d1[-(1:(len3))],log(d2[-(1:(len3))]),xlim=c(min(d1),max(d1)),ylim=c(min(log(d2[sp[1]:n])),max(log(d2[sp[1]:(sp[1]+cp2-1)]))),xlab="S",ylab="Log(Intensity)",pch=1,col="blue",cex=0.75)
    lines(d1[min(sp):(len3)],alpha_curve2)
    text((d1[n]-xspace*4),max(log(d2[c(1,n)])-yspace*20),bquote(hat(R)[g] == .(rg)))
    text((d1[n]-xspace*3.1),(max(log(d2[c(1,n)]))-yspace*26),paste("Std. Deviation = ",serg,sep=""))
    points(d1[sp[1]:(len3)],log(d2[sp[1]:(len3)]),col="red",pch=20,cex=1/nreps)
    for(i in 2:nreps){
      d1 = M[,1]
      d2 = M[,i+1]-fit$coef[i]+fit$coef[1]

      points(d1[-(1:(len3))],log(d2[-(1:(len3))]),pch=1,col="blue",cex=0.75)
      points(d1[sp[i]:(len3)],log(d2[sp[i]:(len3)]),col="red",pch=20,cex=1/nreps)
      if(sp[i] > 1){
        points(d1[1:(sp[i]-1)],log(d2[1:(sp[i]-1)]),pch=1,col="blue",cex=0.75)  
      }
    }
    lines(d1[min(sp):(len3)],alpha_curve2,lwd=1.7)
    legend("topright", c("Data points used to fit curve","Fitted curve","Bias-variance criterion range"),cex=.8,pch=c(20,NA,NA), col=c("red","black","black"), lty=0:2);
    lines(d1[min(sp):(min(sp)+cp2-1)],rep(max(log(d2[-(1:sp[1])]))+.1,cp2),lwd=2,lt=2)
  }
}
  #return Rg and its standard devation
  return(c(output[1],output[2],t,cp2))
}


