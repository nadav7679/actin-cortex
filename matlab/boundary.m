function [pl, ql, pr, qr] = boundary(xl, ul, xr, ur, t)
%  Boundary conditions

k_s = 1;
k_gr = 8.7;
k_br = 2.16e-5;
a_br = 2;
a_gr = 2;
l = 0.003;

r_p = (k_br*ul(2)^2 + k_gr*ul(2))/(a_br*k_br*ul(2)^2 + a_gr*k_gr*ul(2));
r_m_flux = k_br*ul(1)*ul(2)^2 + k_gr*ul(1)*ul(2);

pl = [ul(1) - l*r_p; r_m_flux; 0; ul(4); ul(5) - ul(1)];
ql = [0; 1; 1; 0; 0];
pr = [0; 0; 0; 0; ur(5) - ul(1)];
qr = [1; 1; 1; 1; 0];

end