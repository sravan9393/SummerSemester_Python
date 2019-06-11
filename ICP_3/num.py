import numpy as np

list = np.random.randint(low=1, high=20, size=15)


print("-----------Before modification---------\n")
print(list)

print("\n-----------After Modification----------\n")
result = np.where(list == np.amax(list))

for index in result[0]:
    list[index] = 0

print(list)
