import numpy as np
def dfs(demo_mat,target,index,visited):
    m, n = demo_mat.shape
    if index == len(target):
        return True
    r,c = visited.pop()
    res = []
    if c > 0:
        res.append((r,c-1))
    if c < n-1:
        res.append((r,c+1))
    if r > 0:
        res.append((r-1,c))
    if r < m-1:
        res.append((r+1,c))
    for r,c in res:
        if (r,c) not  in visited and demo_mat[r][c]==target[index]:
            visited.add((r,c))
            if dfs(demo_mat,target,index+1,visited):
                return True
    return False
def check(demo_mat,target):
    m,n = demo_mat.shape
    fis_char = target[0]
    for i in range(n):
        if demo_mat[0][i] == fis_char and dfs(demo_mat,target,1,set([(0,i)])):
            return True
        if demo_mat[m-1][i] == fis_char and dfs(demo_mat,target,1,set([(m-1,i)])):
            return True
    for i in range(1,m-1):
        if demo_mat[i][0] == fis_char and dfs(demo_mat,target,1,set([(i,0)])):
            return True
        if demo_mat[i][n-1] == fis_char and dfs(demo_mat,target,1,set([(i,n-1)])):
            return True
    return False
if __name__ == '__main__':
    demo_mat = np.array([['A','B','C'],['A','B','C'],['A','B','C'],['A','B','C']])
    target = 'ABCD'
    print(check(demo_mat,target))


