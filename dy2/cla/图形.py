def check_string_in_matrix(string, matrix):
    m, n = len(matrix), len(matrix[0])
    # 遍历第一行和最后一行的元素
    for i in range(n):
        if matrix[0][i] == string[0] and dfs(matrix, string, 1, set([(0, i)])):
            return True
        if matrix[m-1][i] == string[0] and dfs(matrix, string, 1, set([(m-1, i)])):
            return True
    # 遍历第一列和最后一列的元素
    for i in range(1, m-1):
        if matrix[i][0] == string[0] and dfs(matrix, string, 1, set([(i, 0)])):
            return True
        if matrix[i][n-1] == string[0] and dfs(matrix, string, 1, set([(i, n-1)])):
            return True
    return False

# 使用深度优先搜索判断当前位置是否与字符串中下一个字符相邻
def dfs(matrix, string, index, visited):
    if index == len(string):  # 字符串所有字符都找到了匹配的位置
        return True
    m, n = len(matrix), len(matrix[0])
    r, c = visited.pop()
    adjacents = []
    if r > 0:
        adjacents.append((r-1, c))
    if r < m-1:
        adjacents.append((r+1, c))
    if c > 0:
        adjacents.append((r, c-1))
    if c < n-1:
        adjacents.append((r, c+1))
    for r, c in adjacents:
        if (r, c) not in visited and matrix[r][c] == string[index]:
            visited.add((r, c))
            if dfs(matrix, string, index+1, visited):
                return True
    return False

# 测试
matrix = [['A', 'B', 'C'], ['A', 'B', 'C'], ['A', 'B', 'C'], ['A', 'B', 'C']]
string_to_check = 'ABCD'

result = check_string_in_matrix(string_to_check, matrix)

if result:
    print(f"字符串 '{string_to_check}' 符合规则，在矩阵中找到了匹配的字符串")
else:
    print(f"字符串 '{string_to_check}' 不符合规则，或不存在于矩阵中")