import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

latitude = [
    27.657999999999994, 26.979999999999997, 26.296999999999997, 25.296999999999997, 
    25.764000000000006, 26.764000000000006, 27.764000000000006, 27.995000000000022, 
    26.995000000000022, 25.995000000000022, 24.995000000000022, 23.995000000000022, 
    22.995000000000022, 21.995000000000022, 22.995000000000022, 21.995000000000022, 
    22.995000000000022, 21.995000000000022, 22.022999999999996, 21.389999999999997, 
    21.761999999999993, 21.68299999999999, 22.50699999999999, 21.50699999999999, 
    22.50699999999999, 23.50699999999999, 23.534000000000006, 23.542000000000012, 
    23.414000000000005, 23.77900000000001, 23.476999999999997, 23.78600000000001, 
    23.476999999999997, 23.647000000000006, 23.624000000000002, 23.79400000000001, 
    23.567000000000004, 23.88500000000001, 23.633000000000003, 23.413, 
    23.095, 23.739999999999995, 23.432000000000002, 24.162000000000006, 
    23.254, 24.254, 23.937, 24.937, 25.31200000000001, 24.31200000000001, 
    23.32900000000001
]

longitude = [
    -78.0, -77.0, -76.0, -75.0, -74.0, -73.0, -72.0, -71.0, -70.0, -69.0, 
    -68.0, -67.0, -66.0, -65.0, -64.0, -63.0, -62.0, -61.0, -60.0, -59.0, 
    -58.0, -57.0, -56.0, -55.0, -54.0, -53.0, -52.0, -53.0, -52.0, -53.0, 
    -52.0, -53.0, -52.0, -53.0, -52.0, -53.0, -52.0, -53.0, -52.0, -53.0, 
    -52.0, -53.0, -52.0, -53.0, -52.0, -51.41400000000005, -50.41400000000005, 
    -51.41400000000005, -52.41400000000005, -53.41400000000005, -52.41400000000005
]

# Create a map using Basemap
plt.figure(figsize=(12, 8))
m = Basemap(projection='merc', llcrnrlat=10, urcrnrlat=40, llcrnrlon=-90, urcrnrlon=-40, resolution='i')
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary()
m.drawparallels(range(10, 41, 5), labels=[1,0,0,0])
m.drawmeridians(range(-90, -39, 10), labels=[0,0,0,1])

x, y = m(longitude, latitude)

m.plot(x, y, marker='o', linestyle='-', color='b', markersize=5)

plt.title('Hurricane Path Trace on World Map')
plt.show()
