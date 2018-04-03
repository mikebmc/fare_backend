import overpy

api = overpy.Overpass()
result = api.query('node["name"="Gielgen"];out body;')

print('len(result.nodes) =', len(result.nodes))
print('len(result.ways) =', len(result.ways))
print('len(result.relations) =', len(result.relations))

print('result.nodes[0].tags =', result.nodes[0].tags)
