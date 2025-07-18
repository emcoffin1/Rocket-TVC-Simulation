# Rocket-TVC-Simulation
The purpose of this document is to simulate the flight of 
a bi-propellant, thrust vector controlled rocket to the highest fidelity that can be accomplished.

The full simulation uses various models to accurately simulate vehicle performance, environmental
conditions, and other phenomenon such as the coriolis affect and change in gravity over flight. RK4 
integration is used to approximate the changes in state variables between each time step. 

A full engine simulation has been created to simulated flow of fluid from the ullage tanks to the flow
of the exhaust upon nozzle exit. It accounts for a change in mass flow rate and chamber pressure as 
corresponding values using equation supplied in https://www.grc.nasa.gov/www/k-12/airplane/rktthsum.html#.
There are also functions to determine the volume of fuel, liquid oxygen, and nitrogen gas depending on 
flight requirements and tank sizes.

A full atmospheric model is used to simulate the affects of altitude on various flight conditions. 
It uses NRLMSISE-00 (via pymsis) to acquire highly accurate air density and temperature, which is then used to determine
static pressure and other variables. 

[WIP]
Various models are still being implemented such as a high fidelity aerodynamics model, a thrust 
vector control model, structural dynamics model, and more.

[RUNNING]
To run the program, the current file to run is test_rocket.py. This will initialize all values, 
and allows manual adjustment of pyplots or printing.


