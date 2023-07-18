function [c, f, s] = equations(x, t, u, dudx)
% u is a vector of the following: (p, m, c_f, c_b, probe)
alpha = 0.01;
beta = 0.7;
Dm = 8.4;
Dc = 10;
k_s = 1;
k_gr = 8.7;
k_br = 2.16e-5;
a_br = 2;
a_gr = 2;
l = 0.003;

disp(size(u(1)))
v = (a_br*k_br*u(2, 1)^2 + a_gr*k_gr*u(2, 1))*u(1, 1); % TODO: make p and m be at x=0
depoly = alpha*u(1)*u(4);
sever = k_s*u(3)*u(1);

c =  [1; 1; 1; 1; 1];
f = [0; Dm*(1-beta*u(1))*dudx(2); Dc*dudx(3); 0; 0];
s = [v*dudx(1)-depoly; depoly; -sever + depoly; v*dudx(4) + sever - depoly; 0];
end