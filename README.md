# Dielectric Stopping Power
Calculates the stopping power of a charged particle with some initial velocity using the dielectric function.
A nice expression for the stopping power can be found in M. D. Barriga-Carrasco, PRE 79, 027401 (2009).
In this formalism, we must calculate a double integral in wavenumber (k) and angular frequency (omega) space of the electron loss function. The electron loss function is just the negative of the imaginary part of the inverse of the dielectric function, which is usually approximated. Here, we use the random phase approximation dielectric function based on the uniform electron gas. We can improve upon this by using the Mermin dielectric function, which includes some information of the influence of the ions on the electronic response.
