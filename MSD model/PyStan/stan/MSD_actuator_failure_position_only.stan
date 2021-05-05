functions{
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
    matrix[2,N] u; // 2d input -- IN THIS CASE, THERE ARE TWO SEPARATE ACTUATORS
    real T; // timestep length!
}
parameters {
    // initial system parameters
    real<lower=0.0> m;       // mass parameter
    real<lower=0.0> k;       // spring parameter
    real<lower=0.0> b;       // damper parameter
    vector<lower=0.0>[D] r;  // measurement noise stds
    vector<lower=0.0>[O] q;  // process noise stds
    // switch parameters
    real<lower=0.0,upper=1.0> s1; // fault in actuator 1
    real<lower=0.0,upper=1.0> s2; // fault in actuator 2
    real<lower=1.0,upper=N> t; // time of change
    matrix[O,N] z;           // states
}
model {
    vector[2] z_inter;
    int before;
    real delt;
    // noise stds priors (i think these will draw them from the )
    r ~ cauchy(0, 1.0);
    q[1] ~ cauchy(0, 1.0); // noise on each state assumed independant
    q[2] ~ cauchy(0, 1.0);
    // prior on parameters
    m ~ normal(2.0, 0.1);
    k ~ normal(0.25, 0.1);
    b ~ normal(0.7, 0.1);

    t ~ uniform(1,N); // prior is uniform over the window
    s1 ~ uniform(0.0,1.0); // incredible quality
    s2 ~ uniform(0.0,1.0); // prior knowledge

    before = floor_search(t,1,N); // should return an integer, stan doesn't allow real -> int conversion
    delt = (t - floor(t))*T; // the timestep within the update 

    // initial state prior
    z[1,1] ~ normal(3,0.05); // well informed 
    z[2,1] ~ normal(0,0.05); // small prior on velocity (going to start the sim with zero speed every time)
   
    // state likelihood (apparently much better to do univariate sampling twice)
    z[1,2:before] ~ normal(z[1,1:before-1] + T*z[2,1:before-1], q[1]);
    z[2,2:before] ~ normal(z[2,1:before-1] + -(k*T/m)*z[1,1:before-1] + -(b*T/m)*z[2,1:before-1] + (T/m)*(u[1,1:before-1]+u[2,1:before-1]), q[2]); // input affects second state only
    z_inter[1] = z[1,before] + delt*z[2,before];
    z_inter[2] = z[2,before] + -(k*delt/m)*z[1,before] + -(b*delt/m)*z[2,before] + (delt/m)*(u[1,before]+u[2,before]);
    z[1,before+1] ~ normal(z_inter[1] + (T - delt)*z_inter[2], (delt/T)*q[1] + (T-delt)*q[1]/T);
    z[2,before+1] ~ normal(z_inter[2] + -(k*(T-delt)/m)*z_inter[1] + -(b*(T - delt)/m)*z_inter[2] + ((T - delt)/m)*(s1*u[1,before] + s2*u[2,before]), (delt/T)*q[2] + (T-delt)*q[2]/T); // input affects second state only
    z[1,before+2:N] ~ normal(z[1,before+1:N-1] + T*z[2,before+1:N-1], q[1]);
    z[2,before+2:N] ~ normal(z[2,before+1:N-1] + -(k*T/m)*z[1,before+1:N-1] + -(b*T/m)*z[2,before+1:N-1] + (T/m)*(s1*u[1,before+1:N-1] + s2*u[2,before+1:N-1]), q[2]); // input affects second state only
    
    // measurement likelihood
    y[1,1:before] ~ normal(z[1,1:before], r[1]); // measurement of first state only
    // y[2,1:before] ~ normal(-(k/m)*z[1,1:before] - (b/m)*z[2,1:before] + (u[1,1:before]+u[2,1:before])/m, r[2]); // acceleration measurement?
    y[1,before+1:N] ~ normal(z[1,before+1:N], r[1]); // measurement of first state only
    // y[2,before+1:N] ~ normal(-(k/m)*z[1,before+1:N] - (b/m)*z[2,before+1:N] + (s1*u[1,before+1:N] + s2*u[2,before+1:N])/m, r[2]); // acceleration measurement?
}