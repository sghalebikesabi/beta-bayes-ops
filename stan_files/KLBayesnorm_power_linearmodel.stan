
data {
   
   int<lower=0> n;// number of observations
   int<lower=0> p;// number of predictors (inc. intercept)
   vector[n] y;// response
   matrix[n,p] X;// predictors
   real mu_0;// prior mean for theta
   real<lower=0> v_0;// prior scale multiplier for theta
   real<lower=0> a_0;// shape of IG prior on sigma2
   real<lower=0> b_0;// scale of IG prior on sigma2
   real<lower=0> w;// Gibbs posterior weight
}

parameters {
   
   vector[p] theta;// regression parameters
   real<lower = 0> sigma2;// residual variance

}

transformed parameters {
   vector[n] lin_pred;
   lin_pred = X*theta;

}

model {

   target += inv_gamma_lpdf(sigma2 | a_0, b_0);
   target += normal_lpdf(theta | mu_0, sqrt(sigma2*v_0));
   /*
   for(i in 1:n){
      y[i] ~ normal(mu, sqrt(sigma2));
   }
   */
   target += w * normal_lpdf(y | lin_pred, sqrt(sigma2));
   
}

