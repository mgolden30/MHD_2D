%{
Derive the curl of the MHD equations symbolically
%}

clear;

%Work with potentials to enforce divergenceless
syms psi(x,y) A(x,y)

vx= diff(psi, y);
vy=-diff(psi, x);

Bx= diff(A, y);
By=-diff(A, x);

%ignore the trivial bits
dvxdt = -vx*diff(vx,x) - vy*diff(vx, y) + Bx*diff(Bx,x) + By*diff(Bx, y)
dvydt = -vx*diff(vy,x) - vy*diff(vy, y) + Bx*diff(By,x) + By*diff(By, y)

dBxdt = -vx*diff(Bx,x) - vy*diff(Bx, y) + Bx*diff(vx,x) + By*diff(vx, y)
dBydt = -vx*diff(By,x) - vy*diff(By, y) + Bx*diff(vy,x) + By*diff(vy, y)

j  = diff(By,x) - diff(Bx,y)
om = diff(vy,x) - diff(vx,y)


%%EQUATION 1: momentum
eq1  = diff(dvydt, x) - diff(dvxdt, y);
cl1  = -vx*diff(om,x) - vy*diff(om, y)+ Bx*diff(j,x) + By*diff(j, y);
err1 = simplify( eq1 - cl1 )

%%EQUATION 2: induction
eq2  = diff(dBydt, x) - diff(dBxdt, y);
cl2  = -vx*diff(j,x) - vy*diff(j, y) + Bx*diff(om,x) + By*diff(om, y);
err2 = simplify( eq2 - cl2 )

%% curl of other vectors
sx = diff(A,x) * diff(psi,x,y) + diff(A,y) * diff(psi,y,y);
sy = diff(A,y) * diff(psi,x,y) + diff(A,x) * diff(psi,x,x);

curl = simplify(diff(sx,x) - diff(sy,y));

err3 = err2 - 2*curl

%% Another way of doing it
tx = diff(vx,x)*Bx + diff(vy,x)*By;
ty = diff(vx,y)*Bx + diff(vy,y)*By;

curl = simplify( diff(ty,x) - diff(tx,y) )

err2 - 2*curl