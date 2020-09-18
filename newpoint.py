# New point

point = [float(x) for x in input('Longitude Latitude Elevation:').split()]
point.append(1.367)

point = point*2
region = []
if point[0] >=90 and point[1]>25 :
    region.append(1)
if point[1]>=30 and point[0]<90:
    region.append(1)
if point[1]<30 and point[0]<90:
    region.append(2)
