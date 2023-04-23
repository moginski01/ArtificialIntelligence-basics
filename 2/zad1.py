import numpy as np

x_in = 2
weight = 0.5
alpha = 0.1

expected_weight = 0.8
error= 0

def predict(inp_vec, weight):
    return inp_vec*weight;
#dla 5 przejsc
for i in range(4):
    delta = 2*(predict(x_in,weight)-expected_weight)*x_in
    weight = weight-(delta*alpha)
error = (predict(x_in,weight)-expected_weight)**2

print("5 Przejść")
print("output: "+str(predict(x_in,weight)))
print("error: "+str(error))
print("waga: "+str(weight))


weight=0.5
for i in range(19):
    delta = 2*(predict(x_in,weight)-expected_weight)*x_in
    weight = weight-(delta*alpha)
error = (predict(x_in,weight)-expected_weight)**2
print("20 Przejść")
print("output: "+str(predict(x_in,weight)))
print("error: "+str(error))
print("waga: "+str(weight))



#przy takim wspołczynniku przeskoczyliśmy po prostu za daleko
x_in = 2
alpha = 1
weight=0.5
for i in range(19):
    delta = 2*(predict(x_in,weight)-expected_weight)*x_in
    weight = weight-(delta*alpha)
error = (predict(x_in,weight)-expected_weight)**2
print("20 Przejść")
print("output: "+str(predict(x_in,weight)))
print("error: "+str(error))
print("waga: "+str(weight))