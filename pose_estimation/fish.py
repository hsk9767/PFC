import sys

n, i, m = map(int, sys.stdin.readline().split())
fish = []

for c in range(m):
    fish_x, fish_y = map(int, sys.stdin.readline().split())
    fish.append([fish_x-1, fish_y-1])

sea = [[0 for a in range(n+i)] for j in range(n+i)]

for a in range(m):
   sea[fish[a][0]+int(i/2)][fish[a][1]+int(i/2)] = 1
fishes = 0   
mask = []

for b in range(int(i/2)-1):
    mask_size=(b+1,int(i/2)-b-1)
    mask.append(mask_size)

for point in fish:
    temp_sea = sea[point[0]:point[0]+i+1][point[1]:point[1]+i+1]
    for mask_size in mask:
      for i in range(len(temp_sea)-mask_size[0]+1):
         for j in range(len(temp_sea)-mask_size[1]+1):
            _temp_sea = temp_sea[i:i+mask_size[0]+1][j:int(j+mask_size[1]+1)]
            _fishes = 0
            for k in range(len(_temp_sea)): 
               _fishes += sum(_temp_sea[k])
            if _fishes > fishes:
               fishes = _fishes
             
print(fishes)