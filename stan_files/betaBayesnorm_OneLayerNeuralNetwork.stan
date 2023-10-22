
functions {
  
  vector hidden_layer(vector h_input, int N_h, matrix weights, vector bias){
    //h_input is the N_{h-1} dimensional vector of inouts
    //N_h is the number of unit in this hidden layer (i.e. the dimension of the output)
    // weights is the N_{h-1} * N_h vector of weights
    // biases is the N_h dimension vector of the biases
    vector[N_h] h_output;
    for(j in 1:N_h){
      h_output[j] = fmax(weights[,j]'*h_input + bias[j],0);
    }
    return h_output;
  }
  
  real output_layer(vector h_input, vector weights, real bias){
    //h_input is the N_{h-1} dimensional vector of inouts
    // weights is the N_{h-1} * N_h vector of weights
    // biases is the N_h dimension vector of the biases
    real h_output;
    h_output = weights'*h_input + bias;
    
    return h_output;
  }
  
}


data {
   
   int<lower=0> n;// number of observations
   int<lower=0> p;// number of predictors (inc. intercept)
   vector[n] y;// response
   matrix[n,p] X;// predictors
   real mu_0;// prior mean for weigths and biases
   real<lower=0> v_0;// prior scale for weigths and biases
   real<lower=0> a_0;// shape of IG prior on sigma2
   real<lower=0> b_0;// scale of IG prior on sigma2
   int<lower=0> N_h;// number of hidden units in hidden layer
   //real<lower=0> w;// Gibbs posterior weight
   real beta_p;// beta-divergence parameter
   real<lower=0> sigma2_lower;// lower bound on the residual variance
   
}

parameters {
   
   matrix[p,N_h] w_input;
   vector[N_h] b_input;
   vector[N_h] w_output;
   real b_output;
   real<lower=sigma2_lower> sigma2;

}

transformed parameters {
  
   real int_term;
   // 1/(beta_p + 1)*int f(z; X, beta)^(beta_p+1)dz
   int_term = (1 / ((2.0*pi())^(beta_p / 2.0) * (1 + beta_p)^1.5*(sigma2^(beta_p / 2))));
  
}



model {
  
  // Neural Network
  vector[n] mu;
  for(i in 1:n){
    mu[i] = output_layer(hidden_layer(X[i,]', N_h, w_input, b_input), w_output, b_output);
  }
  
  // Priors 
  target += normal_lpdf(to_vector(w_input) | mu_0, sqrt(v_0));
  target += normal_lpdf(b_input | mu_0, sqrt(v_0));
  target += normal_lpdf(w_output | mu_0, sqrt(v_0));
  target += normal_lpdf(b_output | mu_0, sqrt(v_0));
  target += inv_gamma_lpdf(sigma2 | a_0, b_0);
   
  // betaD-loss  
  for(i in 1:n){
    //target += (w*((1/beta_p) * exp(normal_lpdf(y[i,1] | mu[i], sqrt(sigma2)))^(beta_p) - int_term));
    target += ((1/beta_p) * exp(normal_lpdf(y[i] | mu[i], sqrt(sigma2)))^(beta_p) - int_term);
  }

}
