n = 256;

nu  = 1/100; %fluid viscosity
eta = 1/100; %magnetic viscosity

[x,y] = meshgrid( (0:n-1)/n*2*pi );

om = cos(x+0.3+y) + cos(y-0.2);
j  = cos(3*x).*sin(2*y) + cos(x-0.1);

%Optional forcing fields for flow and magnetic fields
forcing = -4*cos(4*y);
forcingB=  0*cos(3*y);

%Optional mean magnetic fields
mean_Bx = 0.0;
mean_By = 0.1;

M = 512; %number of outputs to save
dt = 0.005;

omega   = zeros(n,n,M);
current = zeros(n,n,M);

draw_every = 16;

colormap bluewhitered


%Define a params struct that holds all of our domain information
params.nu  = nu;
params.eta = eta;
params.forcing = forcing;
params.forcingB = forcingB;
params.mean_Bx = mean_Bx;
params.mean_By = mean_By;

vidObj = VideoWriter('mhd.avi');
vidObj.FrameRate = 15;
open(vidObj);

iteration = 1; %index for writing out

for t = 1:M*draw_every
  t/draw_every %comment this out if you want

  [om, j] = rk4( om, j, dt, params );
  if mod(t, draw_every) ~= 0
    continue;
  end

  omega(:,:,iteration) = om;
  current(:,:,iteration) = j;
  iteration = iteration + 1;  

  plot_state(om, j, params);
  drawnow;
  
  % Write each frame to the file.
  currFrame = getframe(gcf);
  writeVideo(vidObj,currFrame);
end
  
% Close the file.
close(vidObj);

save("mhd_sim.mat", "current", "omega", "dt", "draw_every", "params");


function   plot_state(om, j, params)
  tiledlayout(1,2);

  [vx, vy, Bx, By] = vectors( om, j, params );

  n = size(om,1);
  d = round(n/16); %skip this many
  ss = 1:d:n;

  [xx,yy] = meshgrid(ss);

  %Just eyeballing a colorbar
  scale  = 10; %colorbar
  scale2 = 5;%vector fields
  nexttile
  imagesc(om);
  hold on
  quiver( xx,yy,scale2*vx(ss,ss), scale2*vy(ss,ss), 'off', "LineWidth", 2, "color", "black" )
  hold off
  colorbar
  axis square
  clim([-1 1]*scale);
  title("$\omega$ and $\vec{u}$", "Interpreter", "latex", "FontSize", 32)
  set(gca, 'ydir', 'normal')
  xticks([1,n])
  yticks([1,n])

  nexttile
  imagesc(j);
  hold on
  quiver( xx,yy, scale2*Bx(ss,ss), scale2*By(ss,ss), 'off', "LineWidth", 2, "color", "black" )
  hold off
  colorbar
  axis square
  clim([-1 1]*scale);
  title("$j$ and $\vec{B}$", "Interpreter", "latex", "FontSize", 32)
  set(gca, 'ydir', 'normal')
  xticks([1,n])
  yticks([1,n])

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

function [om, j] = rk4( om, j, dt, params )
  [k1, l1] = velocity( om, j, params );
  [k2, l2] = velocity( om + dt*k1/2, j + dt*l1/2, params );
  [k3, l3] = velocity( om + dt*k2/2, j + dt*l2/2, params);
  [k4, l4] = velocity( om + dt*k3  , j + dt*l3, params );

  om = om + dt*(k1 + 2*k2 + 2*k3 + k4)/6;
  j  = j  + dt*(l1 + 2*l2 + 2*l3 + l4)/6;
end

function [v_om, v_j] = velocity( om,j, params )
    [rhs_om, rhs_j] = nonlinear( om, j, params );
    [om_diss, j_diss] = dissipation( om, j );

    v_om = rhs_om + params.nu*om_diss + params.forcing;
    v_j  = rhs_j  + params.eta*j_diss + params.forcingB;

    [v_om, v_j] = dealias( v_om, v_j );
end

function [rhs_om, rhs_j] = nonlinear( om, j, params )
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
  vx =  real(ifft2( 1i*k.'.*omf.*inv_k_sq ));
  vy = -real(ifft2( 1i*k  .*omf.*inv_k_sq ));
  Bx =  real(ifft2( 1i*k.'.*jf .*inv_k_sq )) + params.mean_Bx;
  By = -real(ifft2( 1i*k  .*jf .*inv_k_sq )) + params.mean_By;

  % mixed derivatives
  psi_xy = real(ifft2( k.*k.'.*omf.*inv_k_sq ));
  A_xy   = real(ifft2( k.*k.'.*jf.*inv_k_sq ));
  
  psi_xx = real(ifft2( k.*k.*omf.*inv_k_sq ));
  A_xx = real(ifft2( k.*k.*jf.*inv_k_sq ));
  
  psi_yy = real(ifft2( k.'.*k.'.*omf.*inv_k_sq ));
  A_yy = real(ifft2( k.'.*k.'.*jf.*inv_k_sq ));
  
  %extra nonlinear term for induction equation
  extra = (A_xx - A_yy).*psi_xy - (psi_xx - psi_yy).*A_xy;

  rhs_om = -vx.*om_x - vy.*om_y + Bx.*j_x  + By.*j_y;
  rhs_j  = -vx.*j_x  - vy.*j_y  + Bx.*om_x + By.*om_y + 2*extra;
end

function [vx, vy, Bx, By] = vectors( om, j, params )
  n = size(om,1);

  k = 0:n-1;
  k(k>n/2) = k(k>n/2)- n;

  inv_k_sq = 1./(k.^2+ k.'.^2);
  inv_k_sq(1,1) = 0.0;

  omf = fft2(om);
  jf  = fft2(j);

  %vector field components
  vx =  real(ifft2( 1i*k.'.*omf.*inv_k_sq ));
  vy = -real(ifft2( 1i*k  .*omf.*inv_k_sq ));
  Bx =  real(ifft2( 1i*k.'.*jf .*inv_k_sq )) + params.mean_Bx;
  By = -real(ifft2( 1i*k  .*jf .*inv_k_sq )) + params.mean_By;

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