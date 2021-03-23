functions {
    int floor_search(real x, int min_val, int max_val) {
        // truncate the space (it's okay)
        if (min_val > x)
            return min_val;
        else if (max_val < x) {
            return max_val;
        }
        real y = floor(x);
        int range = max_val - min_val;
        real mid_pt = min_val;
        while (1)  {
            if (range == 0) return mid_pt; // should be cast t integer
            range = (range + 1) / 2;
            mid_pt += y > mid_pt ? range : -range;
        }
        return min_val
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
    real<lower=0.0,upper=N> t; // time of change
    matrix[O,N] z;           // states
}

model {
    vector[2] z_inter;
    int before;
    real delt;
    // noise stds priors (i think these will draw them from the )
    r1 ~ cauchy(0, 1.0);
    q1 ~ cauchy(0, 1.0); // noise on each state assumed independant
    r2 ~ cauchy(0, 1.0);
    q2 ~ cauchy(0, 1.0);
    // prior on parameters
    m1 ~ cauchy(0, 1.0);
    k1 ~ cauchy(0, 1.0);
    b1 ~ cauchy(0, 1.0);
    m2 ~ cauchy(0, 1.0);
    k2 ~ cauchy(0, 1.0);
    b2 ~ cauchy(0, 1.0);
    t ~ uniform(0,N); // prior is uniform over t

    before = floor_search(t,0,N); // should return an integer, stan doesn't allow real -> int conversion
    delt = t - before; // the timestep within the update 

    // initial state prior
    z[1,1] ~ normal(0,5);
    z[2,1] ~ normal(0,0.05); // small prior on velocity (going to start the sim with zero speed every time)
   
    // state likelihood (apparently much better to do univariate sampling twice)
    z[1,2:before] ~ normal(z[1,1:before-1] + T*z[2,1:before-1], q1[1]);
    z[2,2:before] ~ normal(z[2,1:before-1] + -(k1*T/m1)*z[1,1:before-1] + -(b1*T/m1)*z[2,1:before-1] + (T/m1)*u[1,1:before-1], q1[2]); // input affects second state only
    z_inter[1] = z[1,before] + delt*z[2,before];
    z_inter[2] = z[2,before] + -(k1*delt/m1)*z[1,before] + -(b1*delt/m1)*z[2,before] + (delt/m1)*u[1,before];
    z[1,before+1] ~ normal(z_inter[1] + (T - delt)*z_inter[2], delt*q1[1]/T + (T-delt)*q2[1]/T);
    z[2,before+1] ~ normal(z_inter[2] + -(k2*(T-delt)/m2)*z_inter[1] + -(b2*(T - delt)/m2)*z_inter[2] + ((T - delt)/m2)*u[1,before],  delt*q1[2]/T + (T-delt)*q2[2]/T); // input affects second state only
    z[1,before+2:N] ~ normal(z[1,before+1:N-1] + T*z[2,before+1:N-1], q2[1]);
    z[2,before+2:N] ~ normal(z[2,before+1:N-1] + -(k2*T/m2)*z[1,before+1:N-1] + -(b2*T/m2)*z[2,before+1:N-1] + (T/m2)*u[1,before+1:N-1], q2[2]); // input affects second state only
    
    // measurement likelihood
    y[1,1:before] ~ normal(z[1,1:before], r1[1]); // measurement of first state only
    y[2,1:before] ~ normal(-(k1/m1)*z[1,1:before] - (b1/m1)*z[2,1:before] + u[1,1:before]/m1, r1[2]); // acceleration measurement?
    y[1,before+1:N] ~ normal(z[1,before+1:N], r2[1]); // measurement of first state only
    y[2,before+1:N] ~ normal(-(k2/m2)*z[1,before+1:N] - (b2/m2)*z[2,before+1:N] + u[1,before+1:N]/m2, r2[2]); // acceleration measurement?
}