#stoppingpower
Calculates the stopping power of a charged particle with some initial velocity using the dielectric function.
A nice expression for the stopping power can be found in M. D. Barriga-Carrasco, PRE 79, 027401 (2009).
The difficulty is that the stopping power in this formalism is that we must calculate a double integral in wavenumber (k) and angular frequency (omega) space of the electron loss function times some other terms. The electron loss function is just the negative of the imaginary part of the inverse of the dielectric function, which is usually numerically approximated. So there are a bunch of numerical approximations happinenig all at once.

The current files are not complete in that I cannot currently calculate a decent stopping power (when compared to other people's codes).
