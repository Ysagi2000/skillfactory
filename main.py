import random
list_number = list(range(0, 40))
value = random.choice(list_number)
is_black=True
print(value if value < 39 else "00")
is_odd = True if value % 2 == 0 else False

if 0 < value < 12 or 18 < value < 28:
    is_black = True if is_odd else False
else:
    is_black = False if is_odd else True

if value == 0:
    print("zero")
elif value == 39:
    print("double zero")
else:
    print("ODD" if is_odd else "not odd")
    print("BLACK" if is_black else "red")

