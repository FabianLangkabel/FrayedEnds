from frayedends import NBArraytest
import numpy as np


og = NBArraytest((3, 4))
og.fill_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
a = og.to_numpy()
b = og.get_unsafe()
c = og.get_capsule_convoluded()
d = og.get_capsule_simple()

print(a, "\n\n", b, "\n\n", c, "\n\n", d, "\n\n")

og.double_all()

print(a, "\n\n", b, "\n\n", c, "\n\n", d, "\n\n")

print(og.to_numpy())

d[0, 0] = 3
print("set d[0, 0] = 3\n")
print(og.to_numpy())

og.explode()

print(a, "\n\n", b, "\n\n", c, "\n\n", d, "\n\n")

del og

print(a, "\n\n", b, "\n\n", c, "\n\n", d, "\n\n")


