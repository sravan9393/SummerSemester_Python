input = [20,23,25]
output = []
for weights in input:
    output.append(float(weights/2.2))
print(output)

weight=[weights/2.2 for weights in input]
print(weight)
