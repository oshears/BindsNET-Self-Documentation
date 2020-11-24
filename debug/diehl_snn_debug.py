from bindsnet.models import DiehlAndCook2015

n_neurons = 100
exc = 22.5
inh = 120
theta_plus = 0.05
dt = 1.0

# Build network.
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

print(network)
print(network.connections["Ae","Ai"].w)
print(network.connections["Ai","Ae"].w)