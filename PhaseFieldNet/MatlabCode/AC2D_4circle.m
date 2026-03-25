clear;
Nx=128; Ny=128; Lx=2*pi; Ly=2*pi; hx=Lx/Nx; hy=Ly/Ny;
x=linspace(-0.5*Lx+hx,0.5*Lx,Nx);
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny);
[xx,yy]=ndgrid(x,y); epsilon=0.05; Cahn=epsilon^2;
u=tanh((2-sqrt(xx.^2+yy.^2))/(sqrt(2)*epsilon));
p=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
q=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];


u = repmat(u,2,2);
p = repmat(p,1,2);
q = repmat(q,1,2);
x = linspace(-Lx+hx,Lx,2*Nx);
y = linspace(-Ly+hy,Ly,2*Ny);

p2=p.^2; q2=q.^2; [pp2,qq2]=ndgrid(p2,q2);
dt=0.01; T=20; Nt=round(T/dt); ns=Nt/200;

figure(1); clf;
contourf(x,y,real(u'),[0 0]); axis image


axis([x(1) x(2*Nx) y(1) y(2*Ny)])
pause(0.01)
for iter=1:Nt
    u=real(u);
    s_hat=fft2(Cahn*u-dt*(u.^3-3*u));
    v_hat=s_hat./(Cahn+dt*(2+Cahn*(pp2+qq2)));
    u=ifft2(v_hat);
    if (mod(iter,ns)==0)
        contourf(x,y,real(u'),[0 0]);
        axis image
        axis([x(1) x(2*Nx) y(1) y(2*Ny)])
        pause(0.01)
    end
end
