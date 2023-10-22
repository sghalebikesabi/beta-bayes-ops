
data {
   
   int<lower=0> n;// number of observations
   int<lower=0> p;// number of predictors (inc. intercept)
   vector[n] y;// response
   matrix[n,p] X;// predictors
   real mu_0;// prior mean for beta
   real<lower=0> v_0;// prior scale multiplier for beta
   real<lower=0> a_0;// shape of IG prior on sigma2
   real<lower=0> b_0;// scale of IG prior on sigma2
   //real<lower=0> w;// Gibbs posterior weight
   real beta_p;// beta-divergence parameter
   real<lower=0> sigma2_lower;// lower bound on the residual variance
}


parameters {
   
   vector[p] theta;
   real<lower=sigma2_lower> sigma2;

}

transformed parameters {
  
   real int_term;
   vector[n] lin_pred;
   lin_pred = X*theta;
   // 1/(beta_p + 1)*int f(z; X, beta)^(beta_p+1)dz
   int_term = (1 / ((2.0*pi())^(beta_p / 2.0) * (1 + beta_p)^1.5*(sigma2^(beta_p / 2))));
  
}

model {
  
   target += inv_gamma_lpdf(sigma2 | a_0, b_0);
   target += normal_lpdf(theta | mu_0, sqrt(sigma2*v_0));


   for(i in 1:n){
      //target += (w*((1/beta_p) * exp(normal_lpdf(y[i,1] | lin_pred[i, 1], sqrt(sigma2)))^(beta_p) - int_term));
      target += ((1/beta_p) * exp(normal_lpdf(y[i] | lin_pred[i], sqrt(sigma2)))^(beta_p) - int_term);
   }

}


