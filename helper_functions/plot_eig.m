function plot_eig(D)

figure()
ax = gca;
plot(real(diag(D)),imag(diag(D)),'r*') %   Plot real and imaginary parts
xlabel('Real')
ylabel('Imaginary')
t1 = ['Eigenvalues of a random matrix of dimension ' num2str(size(D,1))];
title(t1)
makeNiceFigure(ax)
