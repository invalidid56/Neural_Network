card = list(range(0, 6))
case = 0
for i in card:
    for l in card:
        if l is not i:
            left = [c for c in card if (c is not l and c is not i)and(c is 0 or c is 5)]
            case += len(left)
print(case)
