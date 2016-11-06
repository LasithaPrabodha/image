from __future__ import division
from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from scipy import optimize
import ctypes
import colorCodes

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    weight : float
        The absolute difference of the mean color between node `dst` and `n`.
    """

    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return diff


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])

def load_image():
    ld_image = cv2.imread('12.jpg',1)
    img_size = Image.open('12.jpg')
    return ld_image,img_size

imag,img_for_size = load_image()
img = cv2.fastNlMeansDenoisingColored(imag,None,10,10,7,21)
#show_img(img)
width,height = img_for_size.size
print("height is ", height)
print("width is " , width)


labels = segmentation.slic(img, compactness=30, n_segments=400)
g = graph.rag_mean_color(img, labels)

seg_image = color.label2rgb(labels, img, kind='avg')
#Write the data to the segmentation.png
cv2.imwrite('segmentation.png',seg_image)
seg_image = segmentation.mark_boundaries(seg_image, labels, (0, 0, 0))

threshold_value = 50
labels2 = graph.merge_hierarchical(labels, g, thresh=threshold_value, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)

dat = []
region_rgb_value = {}

#w, h = 512, 512

#img.show()

#Get region count and the region RGB values
def count_regions(labelNo):
    del dat[:]
    #region_rgb_value.clear()
    region_count = 0
    for x in range(height):
        for y in range(width):
            if(labelNo[x][y] not in dat):
                region_count = region_count +1
                dat.append(labelNo[x][y])
#                b = img[x][y][0]
#                g = img[x][y][1]
#                r = img[x][y][2]
#                colors = r,g,b
#                region_rgb_value[labelNo[x][y]] = colorCodes.get_color_code(colors)
    return region_count
            
def customize_threshold_value(reg_count,thresh_val):
    labels2 = graph.merge_hierarchical(labels, g, thresh=thresh_val, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)
    while not (reg_count <= 15):
        thresh_val = 2 + thresh_val
        print("Thrshold value is",thresh_val)
        labels2 = graph.merge_hierarchical(labels, g, thresh=thresh_val, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)
        reg_count = count_regions(labels2)
    return labels2,reg_count

no_of_regions = count_regions(labels2)
print(no_of_regions)
labels2 , reg =customize_threshold_value(no_of_regions,threshold_value)
print(reg)

g2 = graph.rag_mean_color(img, labels2)

out = color.label2rgb(labels2, img, kind='avg')
cv2.imwrite('formation.png',out)
#Write the data to the formation.png
im_data2 = np.zeros((height, width, 3), dtype=np.uint8)

#region_rgb_value.clear()
tempRegionNo = []

for i in range (height): #traverses through height of the image
    for j in range (width): #traverses through width of the image
        a,b,c = out[i][j]
        colors = a,b,c
        im_data2[i, j] = [a,b,c]
        if(labels2[i][j] not in tempRegionNo):
            tempRegionNo.append(labels2[i][j])
            region_rgb_value[labels2[i][j]] = colorCodes.get_color_code(colors)
            
out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))

print(region_rgb_value)

no_of_pixels_in_region = dict()  
region = {} #This contents all the pixel coordinates of each region  

#find pixel ratio of each region       
for ite in range(reg):
    pix_count = 0 
    for x in range(height):
        for y in range(width):
            if(labels2[x][y] == ite):
                pix_count = pix_count+1
                if(pix_count == 1):
                    region[labels2[x][y]] = [[x,y]]
                else:
                    region[labels2[x][y]].append([x,y])
    no_of_pixels_in_region[ite] = round(pix_count/(height*width),3)
    
coordinateVal = {}
region_cut_points = {}

for ite in range(reg):
    count=0
    for data in region[ite]:
        valX = data[1]
        valY = data[0]
        if(count==0):
            minX=valX
            maxX=valX
            minY=valY
            maxY=valY
            count+=1
        if(minX >= valX):
            minX = valX
        if(maxX <= valX):
            maxX = valX
        if(minY >= valY):
            minY = valY
        if(maxY <= valY):
            maxY = valY
    #Used for identifying the position of the object's X cordinate and Y coordinate
    if(minX <= (width/3)):
        if(maxX <= (width/3)):
            positionX = 2.5
        elif(maxX <= ((2*width)/3)):
            positionX = 5
        else:
            positionX = 10
    elif(minX <= ((2*width)/3)):
        if(maxX <= ((2*width)/3)):
            positionX = 5
        else:
            positionX = 7.5
    else:
        positionX = 7.5
    
    if(minY <= (height/3)):
        if(maxY <= (height/3)):
            positionY = 2.5
        elif(maxY <= ((2*height)/3)):
            positionY = 5
        else:
            positionY = 10
    elif(minY <= ((2*height)/3)):
        if(maxY <= ((2*height)/3)):
            positionY = 5
        else:
            positionY = 7.5
    else:
        positionY = 7.5
    #For the boundary box
    middleX = (maxX-minX)/2
    middleY = (maxY-minY)/2
    
    upLeftCoordinate = minX,minY
    upRightCoordinate = maxX,minY
    bottomLeftCoordinate = minX,maxY
    bottomRightCoordinate = maxX,maxY
    topMiddleCoordinate = int(round(minX+((maxX-minX)/2))),minY
    bottomMiddleCoordinate = int(round(minX+((maxX-minX)/2))),maxY
    leftMiddleCoordinate = minX,int(round(minY+((maxY-minY)/2)))
    rightMiddleCoordinate = maxX,int(round(minY+((maxY-minY)/2)))
    
    #print("Up left :",upLeftCoordinate , "upRight :" , upRightCoordinate , "botleft :", topMiddleCoordinate, "botright :", bottomRightCoordinate)
    
    topMiddleCoordinateX = topMiddleCoordinate[0]
    topMiddleCoordinateY = topMiddleCoordinate[1]
    for data in region[ite]:
        x_val = data[1]
        y_val = data[0]
        if(y_val == topMiddleCoordinateY):
            region_cut_points[ite] = x_val,topMiddleCoordinateY
            
        else:
            topMiddleCoordinateY = topMiddleCoordinateY + 1
            
        
        
    coordinateVal[ite] = ((maxX-minX)/(maxY-minY)),positionX,positionY
    
#merge two dictioanries in order to get the RGB values and pixel count of each region   
main_dic = {}
for key in (region_rgb_value.viewkeys() | no_of_pixels_in_region.keys()):
    if key in region_rgb_value: main_dic.setdefault(key, []).append(region_rgb_value[key])
    if key in no_of_pixels_in_region: main_dic.setdefault(key, []).append(no_of_pixels_in_region[key])
    if key in coordinateVal: main_dic.setdefault(key, []).append(coordinateVal[key])
        
blue = []
green = []
red = []
pixel_ratio = []
rgb_values = []

feature_vector = {}
removable_keys = []

for key,values in main_dic.iteritems():
#    blue.append(values[0][0])
#    green.append(values[0][1])
#    red.append(values[0][2])
#    rgb_values.append(values[0])
#    pixel_ratio.append(values[1])
#    a = values[0]
#    b = values[1]
#    c = values[2]
    
    if(values[1] < 0.05):
        removable_keys.append(key)
    #feature_vector[key] = values[0][0],values[0][1],values[0][2],values[1],round(values[2][0]/width,3),round(values[2][1]/height,3),values[2][2],values[2][3]
    feature_vector[key] = round(values[0]/100,3),values[1],round(values[2][0],3),round(values[2][1]/10,3),round(values[2][2]/10,3)
    
#This is used to remove the data from the dictionary which are having less than 0.05 of pixel ratio
for key in removable_keys:
    feature_vector.pop(key, None)
    
npvec=np.arange(reg)
keys=list(feature_vector.keys())
values=list(feature_vector.values())

mask=np.in1d(keys,npvec)

arr=np.array(keys)[mask]
ori_feature_vector = np.array(values)[mask]

print("removable keyssssssssssssssssssssss")
print(removable_keys)

print("--------ORIGINAL FEATURE VECTOR-------------")
print(ori_feature_vector)

print(region_rgb_value)

cv2.imshow("image1" , seg_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("image2" , out)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("--------------------region_count----------------------------")
print(reg)
#print("--------------------main_dic----------------------------")
#print(main_dic)
#print("--------------------region----------------------------")
#print(region[0])
#print("--------------------coordinateVal----------------------------")
#print(coordinateVal)

#sys.exit("During testing period")


class Neural_Network(object):
    def __init__(self, Lambda):        
        #Define Hyperparameters
        self.inputLayerSize = 5
        self.outputLayerSize = 1
        self.hiddenLayerSize1 = 3
        self.hiddenLayerSize2 = 3

        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize1)
        
        self.W2 = np.random.randn(self.hiddenLayerSize1,self.hiddenLayerSize2)
        
        self.W3 = np.random.randn(self.hiddenLayerSize2, self.outputLayerSize)

        #Regularization Parameter:
        self.lambd = Lambda
        
    def forward(self, X):
        #Propogate inputs though network

        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.sigmoid(self.z3)
        #yHat = self.a3
        self.z4 = np.dot(self.a3, self.W3)
        yHat = self.sigmoid(self.z4) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)


    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.lambd/2)*(sum(self.W1**2)+sum(self.W2**2)+sum(self.W3**2))
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat =self.forward(X)
        delta4 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z4))
        dJdW3 = np.dot(self.a3.T,delta4)/X.shape[0]+ self.lambd*self.W3

        delta3 = np.dot(delta4,self.W3.T) * self.sigmoidPrime(self.z3)
        dJdW2 = np.dot(self.a2.T,delta3)/X.shape[0] + self.lambd*self.W2
        
        delta2 = np.dot(delta3,self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T,delta2)/X.shape[0] + self.lambd*self.W1
        
        return dJdW1, dJdW2, dJdW3
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 and W3 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize1 * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize1))
        
        W2_end = W1_end + self.hiddenLayerSize1*self.hiddenLayerSize2
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize1, self.hiddenLayerSize2))
        
        W3_end = W2_end + self.hiddenLayerSize2*self.outputLayerSize
        self.W3 = np.reshape(params[W2_end:W3_end], (self.hiddenLayerSize2, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2, dJdW3 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel(), dJdW3.ravel()))
    
    def readInputFile(self, filePathName):
        #inputFile = np.genfromtxt(filePathName, skip_header=1, missing_values=('???',' ','null'), filling_values=(0, 0, 0))
        data = np.genfromtxt(filePathName, usecols=(0,1,2,3,4,5,6),skip_header=1, comments='X', missing_values=('???',' ','null', None), filling_values=(0, 0, 0, 0))
        #y = np.genfromtxt(filePathName,usecols=(3), skip_header=1, missing_values=('???',' ','null', 'X'), filling_values=(0, 0, 0, 0 ))
        #return inputFile
        return data

##Need to modify trainer class a bit to check testing error during training:
class trainer2(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = np.average(self.N.costFunction(X, y))
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
    def train2(self, trainX, trainY):
        #Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY

        #Make empty list to store training costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 500, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

        finalcost = self.N.costFunction(trainX,trainY);

        print("Final cost")
        print(finalcost)

#class contain the Neural network training function
class trainer(object):
    def __init__(self, N):

        self.N = N
        
        
    #training function
    def train(self, X, y):
        cost1 = np.average(self.N.costFunction(X,y))
        dJdW1, dJdW2, dJdW3 = self.N.costFunctionPrime(X, y)

        scalar = 3
        self.N.W1 = self.N.W1 + scalar * dJdW1
        self.N.W2 = self.N.W2 + scalar * dJdW2
        self.N.W3 = self.N.W3 + scalar * dJdW3
        cost2 = np.average(self.N.costFunction(X,y))
        print("cost 1 and cost 2")
        print (cost1, cost2)
        
        dJdW1,dJdW2, dJdW3 = self.N.costFunctionPrime(X,y)
        scalar = 3
        self.N.W1 = self.N.W1 - scalar * dJdW1
        self.N.W2 = self.N.W2 - scalar * dJdW2
        self.N.W3 = self.N.W3 - scalar * dJdW3
        cost3 = np.average(self.N.costFunction(X,y))
        print("cost 2 and cost 3")
        print (cost2, cost3)

        previousCost = 5

        x_axis =[]
        y_axis =[]
        z_axis =[]

        iteration = 0
        x_axis=np.append(x_axis, np.average(self.N.W1))
        y_axis=np.append(y_axis, np.average(self.N.W2))
        z_axis=np.append(z_axis, np.average(self.N.W3))

        #check if all cost3 values < all cost2 values if so reduce (scalar * weight)
        if (cost3<cost2):
            print("cost3<cost2")
            while(True):
                #print("cost3<cost2")
                dJdW1,dJdW2, dJdW3 = self.N.costFunctionPrime(X,y)
                scalar = 3
                self.N.W1 = self.N.W1 - scalar * dJdW1
                self.N.W2 = self.N.W2 - scalar * dJdW2
                self.N.W3 = self.N.W3 - scalar * dJdW3
                cost3 = np.average(self.N.costFunction(X,y))
                iteration = iteration + 1
                weight = (np.average(self.N.W1)+np.average(self.N.W2)+np.average(self.N.W3))/3
                

              
                x_axis=np.append(x_axis, np.average(self.N.W1))
                y_axis=np.append(y_axis, iteration)
                z_axis=np.append(z_axis, cost3)
                            
                if(cost3>previousCost):
                    print("cost3>previousCost")
                    print(cost3,previousCost)
                    
                    print("Weights of exit")
                    print("W1 average")
                    print(np.average(self.N.W1))
                    print("W2 average")
                    print(np.average(self.N.W2))
                    print("W3 average")
                    print(np.average(self.N.W3))
                    break
                
                previousCost = cost3
                
                
                
                

            #check if all cost3 values > all cost2 values if so reduce (scalar * weight)
        elif(cost2<cost3):
            print("cost2<cost3")
            while(True):
                dJdW1,dJdW2, dJdW3 = self.N.costFunctionPrime(X,y)
                scalar = 3
                self.N.W1 = self.N.W1 + scalar * dJdW1
                self.N.W2 = self.N.W2 + scalar * dJdW2
                self.N.W3 = self.N.W3 + scalar * dJdW3
                cost2 = np.average(self.N.costFunction(X,y))
                iteration = iteration + 1
                weight = (np.average(self.N.W1)+np.average(self.N.W2)+np.average(self.N.W3))/3

                x_axis=np.append(x_axis, np.average(self.N.W1))
                y_axis=np.append(y_axis, iteration)
                z_axis=np.append(z_axis, cost2)
                
                if(cost2>previousCost):
                    print("cost2>previousCost")
                    print(cost2,previousCost)

                    print("Weights of exit")
                    print("W1 average")
                    print(np.average(self.N.W1))
                    print("W2 average")
                    print(np.average(self.N.W2))
                    print("W3 average")
                    print(np.average(self.N.W3))
                    
                    break
                
                previousCost = cost2
                
                
                
                
        else:
            print("Error in cost comparison in Training function");
            
        return x_axis, y_axis, z_axis        
            
        
NN = Neural_Network(Lambda=0.00000001)

#Training Data:

#trainX = np.array(([184,207,239,0.511,567,389], [54,68,79,0.018,78,89], [72,106,133,0.13,223,123], [153,136,116,0.339,389,309],[18,62,159,0.449,568,345],[88,109,166,0.174,135,243],[220,193,174,0.199,234,234],[172,197,237,0.475,456,324],[172,199,226,0.051,98,67],[128,174,200,0.244,243,135]), dtype=float)
#trainY = np.array(([1], [2], [3],[4], [1], [3], [4] , [1], [3], [3]), dtype=float)

data = NN.readInputFile("finalDataset.txt")


trainX = data[:, [0,1,2,3,4]]
trainY = data[:, [5]]


trainY = trainY.reshape(-1,1)
#Scaling
trainX = trainX/np.amax(trainX, axis=0)
trainY = trainY/np.amax(trainY, axis=0) #Max test score is 100

T = trainer2(NN)
T.train2(trainX,trainY)

testingX = np.array((ori_feature_vector))
yHat = NN.forward(testingX)

classes = []

for res in yHat:
    sky = abs(round((1/7),2) - res)
    sea = abs(round((2/7),2) - res)
    sands = abs(round((3/7),2) - res)
    mountains = abs(round((4/7),2) - res)
    tree = abs(round((5/7),2) - res)
    sun = abs(round((6/7),2) - res)
    grass = abs(round((7/7),2) - res)
    object_class = "Sky"
    #answer = min(sky,tree,sea,sands)
    min_value = sky
    if(min_value >= tree):
        min_value = tree
        object_class = "Tree"
    if(min_value >= sea):
        min_value = sea
        object_class = "Sea"
    if(min_value >= sands):
        min_value = sands
        object_class = "Sands"
    if(min_value >= mountains):
        min_value = mountains
        object_class = "mountains"
    if(min_value >= sun):
        min_value = sun
        object_class = "Sun"
    if(min_value >= grass):
        min_value = grass
        object_class = "grass"
    print(object_class)
    classes.append(object_class)
    #ctypes.windll.user32.MessageBoxA(0, object_class, "The scenary has ", 1)
    
    
    

print("YHat")
print(yHat)
#
#x = np.resize(x, 1000)
#y = np.resize(y, 1000)
#z = np.resize(z, 1000)
#
#
##wireframe plot
#fig = plt.figure()
#ax1 = fig.add_subplot(111, projection='3d')
#
#
#ax1.plot_wireframe(x,y,z)
#
#ax1.set_xlabel('W1 Values')
#ax1.set_ylabel('W2 Values')
#ax1.set_zlabel('W3 Values')
#
#plt.show()

#print("--------------------Data----------------------------")
#print(dat)
#print("--------------------region_rgb_value----------------------------")
#print(region_rgb_value)
#print("--------------------no_of_pixels_in_region----------------------------")
#print(no_of_pixels_in_region)
print("--------------------region_count----------------------------")
print(reg)
#print("--------------------main_dic----------------------------")
#print(main_dic)
#print("--------------------region----------------------------")
#print(region[0])
#print("--------------------coordinateVal----------------------------")
#print(coordinateVal)
print(feature_vector)
print("End")

print(classes)
count_classes = []
final_word = 'Image contains : '
original_classes = []

for objects in classes:
    if((objects == 'Sea' or objects == 'Sands') and (0 not in count_classes)):
        final_word = final_word + "Sky, Sea, Sand "
        original_classes = ['Sky' , 'Sea' , 'Sand']
        count_classes.append(0)
    elif(objects == 'Sun' and (1 not in count_classes)):
        final_word = final_word + "Sky, Sea, Sun "
        original_classes = ['Sky' , 'Sea' , 'Sun']
        count_classes.append(1)
    elif((objects == 'Tree' or objects == 'grass') and (2 not in count_classes)):
        final_word = final_word + "Sky, Trees, Grass "
        original_classes = ['Sky' , 'Tree' , 'grass']
        count_classes.append(2)
    elif(objects == 'mountains' and (3 not in count_classes)):
        final_word = final_word + "Sky, Mountain, Grass, Tree "
        original_classes = ['Sky' , 'mountains' , 'grass' , 'Tree']
        count_classes.append(3)
    elif(objects == 'Sky' and (4 not in count_classes)):
        final_word = final_word + "Sky "
        original_classes = ['Sky']
        count_classes.append(4)
    
#    count = count+1
#    object_name = objects
#    if(count == 1):
#        original_classes = object_name
#        final_word = final_word + object_name
#    else:
#        if(object_name not in original_classes):
#            original_classes = object_name
#            final_word = final_word + "," + object_name 
            
#print("Final Word = " + final_word)
print("Original classes-----------------")
print(original_classes)

beach_possibility = 0
forest_possibility = 0
mountain_possibility = 0
sunset_possibility = 0

for objects in original_classes:
    if(objects == "Sea"):
        beach_possibility = beach_possibility + 1
        sunset_possibility = sunset_possibility + 1
    if(objects == "Sky"):
        beach_possibility = beach_possibility + 1
        sunset_possibility = sunset_possibility + 1
        forest_possibility = forest_possibility + 0.4
        mountain_possibility = mountain_possibility + 1
    elif(objects == "Tree"):
        forest_possibility = forest_possibility + 1
        mountain_possibility = mountain_possibility + 1
        beach_possibility = beach_possibility + 0.2
    elif(objects == "Sand"):
        beach_possibility = beach_possibility + 1
        sunset_possibility = sunset_possibility + 0.5
    elif(objects == "mountains"):
        forest_possibility = forest_possibility + 1
        mountain_possibility = mountain_possibility + 1
        beach_possibility = beach_possibility + 0.3
        sunset_possibility = sunset_possibility + 0.1
    elif(objects == "Sun"):
        sunset_possibility = sunset_possibility + 1
        mountain_possibility = mountain_possibility + 0.1
        beach_possibility = beach_possibility + 0.3
    elif(objects == "grass"):
        forest_possibility = forest_possibility + 1
        mountain_possibility = mountain_possibility + 1
    
print(final_word + " (Beach : " + str(round(beach_possibility/(beach_possibility+forest_possibility+mountain_possibility+sunset_possibility)*100 , 3)) + "%" + " Forest : "  + str(round(forest_possibility/(beach_possibility+forest_possibility+mountain_possibility+sunset_possibility) *100, 3)) + "%"  +" Mountain : " + str(round(mountain_possibility/(beach_possibility+forest_possibility+mountain_possibility+sunset_possibility)*100 , 3)) + "%"  + " Sunset : " + str(round(sunset_possibility/(beach_possibility+forest_possibility+mountain_possibility+sunset_possibility)*100 , 3)) + "%)")   

    
    
    


