import numpy as np
def dfs(demo_mat,target_s,index,visited):
    if index == len(target_s):
        return True
    m,n = demo_mat.shape
    r,c = visited.pop()
    adjian = []
    if c > 0 :
        adjian.append((r,c-1))
    if c < n-1:
        adjian.append((r,c+1))
    if r > 0 :
        adjian.append((r-1,c))
    if r < m-1:
        adjian.append((r+1,c))
    for r,c in adjian:
        if (r,c) not in visited and demo_mat[r][c] == target_s[index]:
            visited.add((r,c))
            if dfs(demo_mat,target_s,index+1,visited):
                return True
    return False
def checkk(demo_mat,target_s):
    m,n = demo_mat.shape
    fis = target_s[0]
    for i in range(n):
        if demo_mat[0][i] == fis and dfs(demo_mat,target_s,1,set([(0,i)])):
            return True
        if demo_mat[m-1][i] == fis and dfs(demo_mat,target_s,1,set([(m-1,i)])):
            return True
    for i in range(1,m-1):
        if demo_mat[i][0] == fis and dfs(demo_mat, target_s, 1, set([(i, 0)])):
            return True
        if demo_mat[i][n-1] == fis and dfs(demo_mat, target_s, 1, set([(i, n-1)])):
            return True
    return False
demo_mat = np.array([['A','B','C'],['A','B','C'],['A','B','C'],['A','B','C']])
target_s = 'ABC'
print(checkk(demo_mat,target_s))