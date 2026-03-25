clear;
Nx=128; Ny=128; Nz=128; Lx=1.2; Ly=1.2; Lz=1.2; hx=Lx/Nx; hy=Ly/Ny; hz=Lz/Nz;
x=linspace(-0.5*Lx+hx,0.5*Lx,Nx);
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny);
z=linspace(-0.5*Lz+hz,0.5*Lz,Nz);
[xx,yy,zz]=ndgrid(x,y,z); epsilon=hx; Cahn=epsilon^2;
u=rand(Nx,Ny,Nz)-0.5;
kx=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
ky=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
kz=2*pi/Lz*[0:Nz/2 -Nz/2+1:-1];
k2x = kx.^2; k2y = ky.^2; k2z = kz.^2;
[kxx,kyy,kzz]=ndgrid(k2x,k2y,k2z);
dt=0.01; T=0.5; Nt=round(T/dt); ns=Nt/10;
for iter=1:Nt
    disp(['iteration = ', num2str(iter)])
    u=real(u);
    s_hat=fftn(Cahn*u-dt*(u.^3-3*u));
    v_hat=s_hat./(Cahn+dt*(2+Cahn*(kxx+kyy+kzz)));
    u=ifftn(v_hat);
    if mod(iter, ns) == 0
        if isempty(p1)
            % First-time setup
            p1 = patch(isosurface(xx, yy, zz, real(u), 0.));
            set(p1, 'FaceColor', 'g', 'EdgeColor', 'none');
            daspect([1 1 1]); 
            camlight; lighting flat; % Simplified lighting
            box on; axis image;
            view(45, 45);
        else
            % Update isosurface data
            iso = isosurface(xx, yy, zz, real(u), 0.);
            set(p1, 'Vertices', iso.vertices, 'Faces', iso.faces);
        end
    end
end