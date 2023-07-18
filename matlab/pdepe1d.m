x = linspace(0, 1, 200);
t = linspace(0, 3, 3000);

m = 0;
sol = pdepe(m, @equations, @initial, @boundary, x, t);

p = sol(:, :, 1);
m = sol(:, :, 2);
c_f = sol(:, :, 3);
c_b = sol(:, :, 4);
probe = sol(:, :, 5);
