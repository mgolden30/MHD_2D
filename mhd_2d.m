n = 512;

nu  = 1/1000;
eta = 1/1000;

[x,y] = meshgrid( (0:n-1)/n*2*pi );

om = cos(x+0.3+y) + cos(y-0.2);
j  = cos(3*x).*sin(2*y) + cos(x-0.1);

forcing = 0*cos(4*y);
forcingB= 0*cos(3*y);


M = 1024*8;
dt = 0.005;

draw_every = 16;

colormap jet

for t = 1:M
  t
  [om, j] = rk4( om, j, nu, eta, dt, forcing, forcingB );
  if mod(t, draw_every) ~= 0
    continue;
  end

  plot_state(om, j);
  drawnow;
end

function   plot_state(om, j)
  tiledlayout(1,2);

  nexttile
  imagesc(om);
  colorbar
  axis square
  clim([-1 1]);
  title("vorticity")


  nexttile
  imagesc(j);
  colorbar
  axis square
  clim([-1 1]);
  title("current density")
end

function [om,j] = dealias( om, j )
  n = size(om,1);

  k = 0:n-1;
  k(k>n/2) = k(k>n/2)- n;

  mask = k.^2+ k.'.^2 >= (n/3)^2;

  om = fft2(om);
  j  = fft2(j);

  om(mask) = 0;
  j(mask)  = 0;
  
  om = real(ifft2(om));
  j  = real(ifft2(j));
end

function [om, j] = rk4( om, j, nu, eta, dt, forcing, forcingB )
  [k1, l1] = velocity( om, j, nu, eta, forcing, forcingB );
  [k2, l2] = velocity( om + dt*k1/2, j + dt*l1/2, nu, eta, forcing, forcingB );
  [k3, l3] = velocity( om + dt*k2/2, j + dt*l2/2, nu, eta, forcing, forcingB);
  [k4, l4] = velocity( om + dt*k3  , j + dt*l3, nu, eta, forcing, forcingB );

  om = om + dt*(k1 + 2*k2 + 2*k3 + k4)/6;
  j  = j  + dt*(l1 + 2*l2 + 2*l3 + l4)/6;
end

function [v_om, v_j] = velocity( om,j,nu,eta, forcing, forcingB )
    [rhs_om, rhs_j] = nonlinear( om, j );
    [om_diss, j_diss] = dissipation( om, j );

    v_om = rhs_om + nu*om_diss + forcing;
    v_j  = rhs_j  + eta*j_diss + forcingB;

    [v_om, v_j] = dealias( v_om, v_j );
end

function [rhs_om, rhs_j] = nonlinear( om, j )
  n = size(om,1);

  k = 0:n-1;
  k(k>n/2) = k(k>n/2)- n;

  inv_k_sq = 1./(k.^2+ k.'.^2);
  inv_k_sq(1,1) = 0.0;

  omf = fft2(om);
  jf  = fft2(j);

  %Derivatives
  om_x = real(ifft2( 1i*k  .*omf ));
  om_y = real(ifft2( 1i*k.'.*omf ));
  j_x  = real(ifft2( 1i*k  .*jf ));
  j_y  = real(ifft2( 1i*k.'.*jf ));

  %vector field components
  vx = real(ifft2( 1i*k.'.*omf.*inv_k_sq ));
  vy =-real(ifft2(   1i*k.*omf.*inv_k_sq ));
  Bx  = real(ifft2( 1i*k.' .*jf.*inv_k_sq ));
  By  =-real(ifft2( 1i*k  .*jf.*inv_k_sq ));

  % mixed derivatives
  psi_xy = real(ifft2( k.*k.'.*omf.*inv_k_sq ));
  A_xy = real(ifft2( k.*k.'.*jf.*inv_k_sq ));
  
  psi_xx = real(ifft2( k.*k.*omf.*inv_k_sq ));
  A_xx = real(ifft2( k.*k.*jf.*inv_k_sq ));
  
  psi_yy = real(ifft2( k.'.*k.'.*omf.*inv_k_sq ));
  A_yy = real(ifft2( k.'.*k.'.*jf.*inv_k_sq ));
  
  %extra nonlinear term for induction equation
  extra = (A_xx - A_yy).*psi_xy - (psi_xx - psi_yy).*A_xy;

  rhs_om = -vx.*om_x - vy.*om_y + Bx.*j_x  + By.*j_y;
  rhs_j  = -vx.*j_x  - vy.*j_y  + Bx.*om_x + By.*om_y + 2*extra;
end

function [om_diss, j_diss] = dissipation( om, j )
  n = size(om,1);

  k = 0:n-1;
  k(k>n/2) = k(k>n/2)- n;

  omf = fft2(om);
  jf  = fft2(j);

  k_sq = k.^2 + k'.^2;

  %Derivatives
  om_diss = real(ifft2( -k_sq.*omf ));
  j_diss  = real(ifft2( -k_sq.*jf ));
  
end