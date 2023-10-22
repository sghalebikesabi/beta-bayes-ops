
data {
   int<lower=0> n;
   int<lower=0> p;
   int<lower=-1,upper=1> y[n];
   matrix[n,p] X;
   real mu_0;
   real<lower=0> v_0;
   real<lower=0> w;
}

parameters {

  vector[p] theta;

}

transformed parameters {

  vector[n] lin_pred;
  lin_pred = X*theta;

}

model {

  theta ~ normal(mu_0, sqrt(v_0));

  for(i in 1:n){
    target += w*(0.5*y[i]*lin_pred[i]-log(exp(0.5*lin_pred[i])+exp(-0.5*lin_pred[i])));
  }

}

