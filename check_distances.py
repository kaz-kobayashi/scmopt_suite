import numpy as np

# Direct PyVRP coordinates 
coords_direct = [(0, 0), (100, 0), (0, 100), (-100, 0), (0, -100)]

print('Direct PyVRP uses these distances:')
for i in range(len(coords_direct)):
    for j in range(len(coords_direct)):
        if i != j and i < 3 and j < 3:
            x1, y1 = coords_direct[i]
            x2, y2 = coords_direct[j]
            euclidean = int(np.sqrt((x1-x2)**2 + (y1-y2)**2))
            print(f'Distance [{i}]->[{j}]: Euclidean={euclidean}')

print('\nOur matrix has these values:')
durations = [
    [   0,  141,  100,  141,  100],
    [ 141,    0,  141,  200,  141],
    [ 100,  141,    0,  141,  200],
    [ 141,  200,  141,    0,  141],
    [ 100,  141,  200,  141,    0],
]

for i in range(3):
    for j in range(3):
        if i != j:
            print(f'Duration [{i}]->[{j}]: {durations[i][j]}')
            
print('\nDistance matrix (duration + 300):')
distances = [[d + 300 for d in row] for row in durations]
for i in range(3):
    for j in range(3):
        if i != j:
            print(f'Distance [{i}]->[{j}]: {distances[i][j]}')