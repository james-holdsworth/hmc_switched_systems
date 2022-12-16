A = [0, 1;
     0, 0];
dt = 0.01;
Adt = A * dt;
Ad = expm(Adt)


%% Adding input with a first order hold
A = [0, 1;
     0, 0];
B = [0;
     1];

Aa = [A,         B;
      zeros(1,2),0];

Aadt = Aa * dt;
Aad = expm(Aadt)

%% c2dprocess
Qc = eye(2);
Qc(1,1) = 0;
[Ad, Bd, Qd] = c2dprocess(A,B,Qc,dt) 


%%
function [Ad,Bd,Qd] = c2dprocess(A,B,Q,T) % i assume this is somehow computationally optimal
    [nx,nu] = size(B);
    F = [-A, Q, zeros(nx,nu);
    zeros(nx,nx), A.', zeros(nx,nu);
    zeros(nu,nx), B.', zeros(nu,nu)]*T;
    G = expm(F);
    Ad = G(nx+1:2*nx,nx+1:2*nx).';
    Bd = G(2*nx+1:2*nx+nu,nx+1:2*nx).';
    Qd = Ad*G(1:nx,nx+1:2*nx);
end


