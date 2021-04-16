functions {
    int floor_search(real y, int min_val, int max_val) {
        real x = floor(y); // stan floor function has return type real
        int range = (max_val - min_val + 1)/2;
        int mid_pt = min_val + range;
        int out;
        // find an int that matches x
        while(range > 0) {
            if(x == mid_pt){
                out = mid_pt;
                range = 0;
            } else {
                range =  (range+1)/2; 
                mid_pt = x > mid_pt ? mid_pt + range: mid_pt - range; 
            }
        }
        return out;
    }
}

data {
    int<lower=0> N; // time horizon in steps
    int<lower=0> O; // order / state count
    int<lower=0> D; // number of measurements
    matrix[D,N] y; // 2d measurement
    matrix[1,N] u; // 1d input
    real T; // timestep length!
}
parameters {
    // initial system parameters
    real<lower=0.0> m1;       // mass parameter
    real<lower=0.0> k1;       // spring parameter
    real<lower=0.0> b1;       // damper parameter
    vector<lower=0.0>[D] r1;  // measurement noise stds
    vector<lower=0.0>[O] q1;  // process noise stds
    // system parameters post switch
    real<lower=0.0> m2;       // mass parameter
    real<lower=0.0> k2;       // spring parameter
    real<lower=0.0> b2;       // damper parameter
    vector<lower=0.0>[D] r2;  // measurement noise stds
    vector<lower=0.0>[O] q2;  // process noise stds
    real<lower=1.0,upper=N> t; // time of change
    matrix[O,N] z;           // states
}

model {
    vector[O] z_inter;
    vector[1] u_inter;

    int before;
    real delt;
    // noise stds priors (i think these will draw them from the )
    r1 ~ cauchy(0, 1.0);
    q1 ~ cauchy(0, 1.0); // noise on each state assumed independant
    r2 ~ cauchy(0, 1.0);
    q2 ~ cauchy(0, 1.0);
    // prior on parameters
    m1 ~ normal(2.0, 0.01);
    k1 ~ normal(0.25, 0.02);
    b1 ~ normal(0.75, 0.02);
    m2 ~ normal(3, 1.0); // we think the mass gets bigger
    k2 ~ normal(0.25, 0.02);
    b2 ~ normal(0.75, 0.02);
    t ~ uniform(1,N); // prior is uniform over the window

    before = floor_search(t,1,N); // should return an integer, stan doesn't allow real -> int conversion
    delt = t - floor(t); // the timestep within the update 

    // initial state prior
    z[1,1] ~ normal(3,0.05);
    z[2,1] ~ normal(0,0.05); // small prior on velocity (going to start the sim with zero speed every time)
   
    // state likelihood (apparently much better to do univariate sampling twice)
    z[1,2:before] ~ normal(z[1,1:before-1] + T*z[2,1:before-1], q1[1]);
    z[2,2:before] ~ normal(z[2,1:before-1] + -(k1*T/m1)*z[1,1:before-1] + -(b1*T/m1)*z[2,1:before-1] + (T/m1)*u[1,1:before-1], q1[2]); // input affects second state only
    // special interval
    u_inter[1] = u[1,before-1]*(1 - delt) + (delt)*u[1,before+1];
    z_inter[1] = z[1,before] + delt*z[2,before];
    z_inter[2] = z[2,before] + -(k1*delt/m1)*z[1,before] + -(b1*delt/m1)*z[2,before] + (delt/m1)*u_inter[1];
    z[1,before+1] ~ normal(z_inter[1] + (T - delt)*z_inter[2], (delt/T)*q1[1] + (T-delt)*q2[1]/T);
    z[2,before+1] ~ normal(z_inter[2] + -(k2*(T-delt)/m2)*z_inter[1] + -(b2*(T - delt)/m2)*z_inter[2] + ((T - delt)/m2)*u_inter[1], (delt/T)*q1[2] + (T-delt)*q2[2]/T); // input affects second state only
    // standard simulation
    z[1,before+2:N] ~ normal(z[1,before+1:N-1] + T*z[2,before+1:N-1], q2[1]);
    z[2,before+2:N] ~ normal(z[2,before+1:N-1] + -(k2*T/m2)*z[1,before+1:N-1] + -(b2*T/m2)*z[2,before+1:N-1] + (T/m2)*u[1,before+1:N-1], q2[2]); // input affects second state only
    
    // measurement likelihood
    y[1,1:before] ~ normal(z[1,1:before], r1[1]); // measurement of first state only
    y[2,1:before] ~ normal(-(k1/m1)*z[1,1:before] - (b1/m1)*z[2,1:before] + u[1,1:before]/m1, r1[2]); // acceleration measurement?
    y[1,before+1:N] ~ normal(z[1,before+1:N], r2[1]); // measurement of first state only
    y[2,before+1:N] ~ normal(-(k2/m2)*z[1,before+1:N] - (b2/m2)*z[2,before+1:N] + u[1,before+1:N]/m2, r2[2]); // acceleration measurement?
}