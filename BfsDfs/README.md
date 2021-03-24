1. Breadth First Search atau BFS adalah algoritma rekursif untuk mencari semua simpul dari grafik atau struktur data pohon.

pada kodingan di bawah ini ada function bfs yang menampung graph dan root yang didalam nya terdapat  visited quieue yang menampung set dari data dan setelah itu akan di masukan ke root dan setelah itu queue akan di lakukan pengulangan while yang menampung deque yang merupakan bagian dari queue dan akan di tampilkan data sesuai queue nya dan setelah itu jika tidak ada yang visited akan di tampung dengan jumlah neighbor ang berkunjung

def bfs(graph, root):

    visited, queue = set(), collections.deque([root])
    visited.add(root)

    while queue:

        vertex = queue.popleft()
        print(str(vertex) + " ", end="")

      
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

dan ini untuk menampung graph nya dan akn memunculkan hasil dari bfs
if __name__ == '__main__':
    graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}
    print("Following is bfs: ")
    bfs(graph, 0)

2. Depth First Search atau DFS adalah algoritma rekursif untuk mencari semua simpul dari suatu grafik atau struktur data pohon. Traversal berarti mengunjungi semua node dari sebuah grafik.

pada kodingan di bawah merupakan function untuk menampung graph,start,dan visited none ini untuk membuat visited kosong dan dibawah kodingan terdapat if yang jika visited nya sma dengan none maka visited akan di set dan akan ditambahkanketika kodingan di jalankan

def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

dan pada kodingan di bawah ini akan di start dan akan di perlihat kan graph nya dan juga akan merlakukan pengulangan sampai si kodingan berhenti sesuai graphic yang sudah di siapkan

print(start)

    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited


*dan yang di bawah ini adalah graph nya
graph = {'0': set(['1', '2']),
         '1': set(['0', '3', '4']),
         '2': set(['0']),
         '3': set(['1']),
         '4': set(['2', '3'])}

dfs(graph, '0')




