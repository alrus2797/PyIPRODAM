import sys
import pprint
from scipy import sparse as sp
import numpy as np
import numpy.linalg as LA
from mayavi.mlab import *

pp = pprint.PrettyPrinter(indent=4)


#print (sys.argv)
name = sys.argv[1]
model = open(name,'r')

model.readline()  #OFF  line
properties = model.readline()  # number of points and faces

nPoints, nFaces = map(int,properties.split(' ')[:-1]) #get that


points = []
faces  = []

#save points
for i in range(nPoints):
    point = map(float,model.readline().split(' '))
    points.append(point)

#save faces
for i in range(nFaces):
    face = map(int,model.readline().split(' ')[1:])
    faces.append(face)

points = np.array(points)
faces = np.array(faces)

def getEquals(f1, f2):
    t1 = f1[:]
    t2 = f2[:]
    t1.sort()
    t2.sort()
    Eq   = []
    Diff = []
    for i in t1:
        for j in t2:
            if i == j: Eq.append(i)
                #break
    #print (Eq)
    if len(Diff) != 2 and len(Eq) != 2:
        return 0,0
    for i in t1:
        if not i in Eq: Diff.append(i)
    for i in t2:
        if not i in Eq: Diff.append(i)
    
    return tuple(Eq), Diff


def getAngleB2V(v1,v2):
    inner = np.inner(v1,v2)
    norm = LA.norm(v1) * LA.norm(v2)
    c = inner / norm
    #print(inner,norm)
    return np.arccos(np.clip(c,-1,1))


#p3 is the origin of p1,p2
def getAngleF3Points(p1,p2,p3):
    #Get 2 vectors (p3 - p1), (p3 - p2)
    v1 = p3 - p1
    v2 = p3 - p2
    #Return the Angle Between v1,v2
    return getAngleB2V(v1,v2)

# v1 = [1,1,0]
# v2 = [1,0,0]
# print("Angle: ", np.degrees(getAngleB2V(v1,v2)))

points = [[0,0,0], [1,0,0], [0,0,1], [0,1,1],
         [1,0,1], [0,1,0], [1,1,1], [1,1,0]]

faces = [[0,2,4], [0,1,4], [2,3,4], [3,4,5], [3,5,6], [0,1,7]]
points = np.array(points)
faces = np.array(faces)
nFaces = len(faces)

Weights = {}
for i in range(nFaces):
    print i, "from" , nFaces
    for j in range(nFaces):
        f1 = faces[i]
        f2 = faces[j]
        Eq, Diff = getEquals(f1,f2)
        if Eq:
            #Points Face 1
            #print (Eq,Diff)
            p1,p2,p3 = points[Eq[0]] , points[Eq[1]] , points[Diff[0]]
            
            #Get angle Between p3-p1 and p3-p2
            angle1 = getAngleF3Points(p1,p2,p3)
            #Get the cotangent
            angle1 = 1.0/np.tan(angle1)

            #Points Face 2
            p1,p2,p3 = points[Eq[0]] , points[Eq[1]] , points[Diff[1]]

            #Get angle Between p3-p1 and p3-p2, but in Face 2
            angle2 = getAngleF3Points(p1,p2,p3)

            #Get the contangent
            angle2 = 1.0/np.tan(angle2)

            #Save the Weight between two Equal points
            Weights[Eq] = (angle1 + angle2) / 2
pp.pprint(Weights)















        












f=figure()
mesh = triangular_mesh(points[:,0],points[:,1],points[:,2],faces,resolution = 100 )
mesh.actor.property.frontface_culling = True
#mesh.actor.property.backface_culling = True
f.scene.renderer.use_depth_peeling=1
f.scene.renderer.maximum_number_of_peels = 0
scene = gcf().scene
scene.renderer.set(use_depth_peeling=True)
#pipeline.surface().
show()