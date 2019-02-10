done = False

def solve(X, Y, solution=[]):
    if not X:
        # yield list(solution)
        sol = list(solution)
        return True, sol
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            ret, sol = solve(X, Y, solution)
            if ret:
                return True, sol
            deselect(X, Y, r, cols)
            solution.pop()
    return False, []

def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols

def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)

def preprocess(X, Y):
    Xnew = {j: set() for j in X}
    for i in Y:
        for j in Y[i]:
            Xnew[j].add(i)
    return Xnew

# def preprocess(X, Y):
#     Xnew = {j: set() for j in X}
#     for i,arr in enumerate(Y):
#         for j in arr:
#             Xnew[j].add(i)
#     return Xnew

if __name__ == "__main__":
    X = [1, 2, 3, 4, 5, 6, 7]

    Y = {
        10: [1, 4, 7],
        20: [1, 4],
        30: [4, 5, 7],
        40: [3, 5, 6],
        50: [2, 3, 6, 7],
        60: [2, 7],
        70: [1,2,3,6] }

    # Y = [
    #     [1, 4, 7],
    #     [1, 4],
    #     [4, 5, 7],
    #     [3, 5, 6],
    #     [2, 3, 6, 7],
    #     [2, 7],
    #     [1, 2, 3, 6]
    #     ]

    X = preprocess(X,Y)

    print (solve(X,Y)[1])

