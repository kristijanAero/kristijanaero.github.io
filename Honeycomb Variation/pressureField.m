c_root = 1.20028; % root chord (m)
c_tip = 0.588;    % tip chord (m)
b = 3.81;         % wing span (m)

cp = @(x) (-0.2068^2 + 0.3243*x - 0.004884) ./ (x.^4 - 1.853^3 + 1.326*x.^2 - 0.01385*x + 0.003363) + 0.1249;
cl = @(y) (-843.3*y.^3 + 3558*y.^2 - 2352*y.^3 + 4514) ./ (y.^4 - 1230*y.^3 + 5627*y.^2 - 4186*y + 7317) + 20.15;

f = @(x, y) cp(x) .* cl(y);

n_y = 30; 
n_x = 10; 

y_values = linspace(0, b, n_y);

x_values = linspace(0, c_root, n_x); 

total_integral = 0;

for i = 1:n_y
    y = y_values(i);
    
    c_y = c_root * (1 - y/b) + c_tip * (y/b);
    
    x_values = linspace(0, c_y, n_x);
    
    f_values = f(x_values, y);
    
    integral_chord = trapz(x_values, f_values);
    
    total_integral = total_integral + integral_chord * (y_values(2) - y_values(1)); % trapezoidal integration over y
end


pressure_data = [];

for i = 1:n_y
    y = y_values(i);
    
    c_y = c_root * (1 - y/b) + c_tip * (y/b);
    
    x_values = linspace(0, c_y, n_x);
    
    f_values = f(x_values, y);
    
    pressure_data = [pressure_data; [x_values', repmat(y, n_x, 1), f_values']];
end

pressure_data(:,3) = pressure_data(:,3)/total_integral;


csvwrite('pressure.csv', pressure_data);

disp('Data has been exported to pressure.csv');
