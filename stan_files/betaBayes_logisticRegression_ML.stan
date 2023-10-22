
data {

   int<lower=0> n;
   int<lower=0> p;
   int<lower=-1,upper=1> y[n];
   matrix[n,p] X;
   real mu_0;
   real<lower=0> v_0;
   real beta_p;

}

parameters {

   vector[p] theta;

}

transformed parameters {

   vector[n] lin_pred;
   lin_pred = X*theta; 

}

model {

   real p_logistic;
   theta ~ normal(mu_0, sqrt(v_0));

   for(i in 1:n){
     p_logistic = (exp(0.5*y[i]*lin_pred[i])/
              (exp(0.5*lin_pred[i])+exp(-0.5*lin_pred[i])));
     
      target += 1/beta_p*p_logistic^beta_p - 
              1/(beta_p+1)*(p_logistic^(beta_p+1)+(1-p_logistic)^(beta_p+1));
   }

}

