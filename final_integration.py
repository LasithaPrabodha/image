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

class Image_anaysis(object):
    
    def __init__(self):        
        #Define Hyperparameters
        self.height = 0
        self.width = 0
    
    def _weight_mean_color(self , graph, src, dst, n):
        diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
        diff = np.linalg.norm(diff)
        return diff
        
    def merge_mean_color(self , graph, src, dst):
        graph.node[dst]['total color'] += graph.node[src]['total color']
        graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
        graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                         graph.node[dst]['pixel count'])
    
    
    def load_image(self):
        ld_image = cv2.imread('images/uploads/'+str(sys.argv[1]),1)
        img_size = Image.open('images/uploads/'+str(sys.argv[1]))
        self.width , self.height = img_size.size
        return ld_image,img_size
       
    def segmentation(self , img):
        labels = segmentation.slic(img, compactness=30, n_segments=400)
        g = graph.rag_mean_color(img, labels)
        
        seg_image = color.label2rgb(labels, img, kind='avg')
        #Write the data to the segmentation.png
        cv2.imwrite('segmentation.png',seg_image)
        seg_image = segmentation.mark_boundaries(seg_image, labels, (0, 0, 0))
        return seg_image , g , labels
        
    def initial_formation(self, labels , g):
        threshold_value = 50
        labels2 = graph.merge_hierarchical(labels, g, thresh=threshold_value, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func= self.merge_mean_color,
                                   weight_func= self._weight_mean_color)
        return labels2
    

    def count_regions(self, labelNo):
        dat = []
        region_count = 0
        for x in range(self.height):
            for y in range(self.width):
                if(labelNo[x][y] not in dat):
                    region_count = region_count +1
                    dat.append(labelNo[x][y])
        return region_count
        
    def customize_threshold_value(self, labels , g , reg_count, thresh_val):
        labels2 = graph.merge_hierarchical(labels, g, thresh=thresh_val, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func= self.merge_mean_color,
                                       weight_func= self._weight_mean_color)
        while not (reg_count <= 15):
            thresh_val = 2 + thresh_val
            print("Thrshold value is",thresh_val)
            labels2 = graph.merge_hierarchical(labels, g, thresh=thresh_val, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func= self.merge_mean_color,
                                       weight_func= self._weight_mean_color)
            reg_count = self.count_regions(labels2)
        return labels2,reg_count
        
    def merge_regions(self , img , labels2):
        g2 = graph.rag_mean_color(img, labels2)
        out = color.label2rgb(labels2, img, kind='avg')
        cv2.imwrite('formation.png',out)
        return out
        
    def extract_color_details(self , out , labels2):
        tempRegionNo = []
        region_rgb_value = {}
        for i in range (self.height): #traverses through height of the image
            for j in range (self.width): #traverses through width of the image
                a,b,c = out[i][j]
                colors = a,b,c
                if(labels2[i][j] not in tempRegionNo):
                    tempRegionNo.append(labels2[i][j])
                    region_rgb_value[labels2[i][j]] = colorCodes.get_color_code(colors)
        return region_rgb_value
                    
    def draw_form_image(self , out , labels2):
        out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
        return out
     
    #find pixel ratio of each region 
    def calculate_pixel_ratio(self , labels2 , reg):
        no_of_pixels_in_region = dict()  
        region = {} #This contents all the pixel coordinates of each region
        for ite in range(reg):
            pix_count = 0 
            for x in range(self.height):
                for y in range(self.width):
                    if(labels2[x][y] == ite):
                        pix_count = pix_count+1
                        if(pix_count == 1):
                            region[labels2[x][y]] = [[x,y]]
                        else:
                            region[labels2[x][y]].append([x,y])
            no_of_pixels_in_region[ite] = round(pix_count/(self.height*self.width),3)
        return region , no_of_pixels_in_region
        
       
    def extract_region_shape_details(self , region):
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
            if(minX <= (self.width/3)):
                if(maxX <= (self.width/3)):
                    positionX = 2.5
                elif(maxX <= ((2*self.width)/3)):
                    positionX = 5
                else:
                    positionX = 10
            elif(minX <= ((2*self.width)/3)):
                if(maxX <= ((2*self.width)/3)):
                    positionX = 5
                else:
                    positionX = 7.5
            else:
                positionX = 7.5
            
            if(minY <= (self.height/3)):
                if(maxY <= (self.height/3)):
                    positionY = 2.5
                elif(maxY <= ((2*self.height)/3)):
                    positionY = 5
                else:
                    positionY = 10
            elif(minY <= ((2*self.height)/3)):
                if(maxY <= ((2*self.height)/3)):
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
        return coordinateVal
            
            
    def merge_feature_set(self , region_rgb_value , no_of_pixels_in_region , coordinateVal):
        main_dic = {}
        for key in (region_rgb_value.viewkeys() | no_of_pixels_in_region.keys()):
            if key in region_rgb_value: main_dic.setdefault(key, []).append(region_rgb_value[key])
            if key in no_of_pixels_in_region: main_dic.setdefault(key, []).append(no_of_pixels_in_region[key])
            if key in coordinateVal: main_dic.setdefault(key, []).append(coordinateVal[key])
        return main_dic
        
        
    def create_feature_vector(self , main_dic):
        feature_vector = {}
        removable_keys = []
        for key,values in main_dic.iteritems():
            if(values[1] < 0.05):
                removable_keys.append(key)
            feature_vector[key] = round(values[0]/100,3),values[1],round(values[2][0],3),round(values[2][1]/10,3),round(values[2][2]/10,3)
        return feature_vector , removable_keys
        
    def create_final_feature_vector(self , feature_vector , removable_keys):
        for key in removable_keys:
            feature_vector.pop(key, None)
        npvec=np.arange(reg)
        keys=list(feature_vector.keys())
        values=list(feature_vector.values())
        mask=np.in1d(keys,npvec)
        arr=np.array(keys)[mask]
        ori_feature_vector = np.array(values)[mask]
        return ori_feature_vector
        
        
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
        """
        Propogate inputs through network
         Parameters
         ----------  
         X : array
             contains the features of an object
        """

        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.sigmoid(self.z3)
        #yHat = self.a3
        self.z4 = np.dot(self.a3, self.W3)
        yHat = self.sigmoid(self.z4) 
        return yHat
        
    def sigmoid(self, z):
        """
        Apply sigmoid activation function to scalar, vector, or matrix
        Parameters
        ----------
        z : array
            metrics to be activated
        """
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        """
        Gradient of sigmoid
        Parameters
        ----------
        z : array
            metrics to be activated
        """
        return np.exp(-z)/((1+np.exp(-z))**2)


    
    def costFunction(self, X, y):
        """
        Compute cost for given X,y, use weights already stored in class.
        Parameters
        ----------
        X : array
            contains the features of an object
        y : array
            contains the output classes
        """
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.lambd/2)*(sum(self.W1**2)+sum(self.W2**2)+sum(self.W3**2))
        return J
        
    def costFunctionPrime(self, X, y):
        """
        Compute derivative with respect to W1,W2 and W3 for a given X and y:
        Parameters
        ----------
        X : array
            contains the features of an object
        y : array
            contains the output classes
        """
        self.yHat =self.forward(X)
        delta4 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z4))
        dJdW3 = np.dot(self.a3.T,delta4)/X.shape[0]+ self.lambd*self.W3

        delta3 = np.dot(delta4,self.W3.T) * self.sigmoidPrime(self.z3)
        dJdW2 = np.dot(self.a2.T,delta3)/X.shape[0] + self.lambd*self.W2
        
        delta2 = np.dot(delta3,self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T,delta2)/X.shape[0] + self.lambd*self.W1
        
        return dJdW1, dJdW2, dJdW3
    
    
    def getParams(self):
        """
        Helper Functions for interacting with other classes:
        """
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel()))
        return params
    
    def setParams(self, params):
        """
        Set W1 and W2 and W3 using single paramater vector.
        Parameters
        ----------
        params : array
        """
        
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
            
class classification(object):
    
    def __init__(self):  
        self.beach_possibility = 0
        self.forest_possibility = 0
        self.mountain_possibility = 0
        self.sunset_possibility = 0
    
    def identify_objects(self , yHat):
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
        return classes
        
    def display_classes(self , classes):
        count_classes = []
        final_word = 'Image contains : '
        original_classes = []
        for objects in classes:
            if((objects == 'Sea' or objects == 'Sands') and (0 not in count_classes)):
                count_classes.append(0)
                if('Sky' not in original_classes):
                    original_classes = original_classes + ['Sky']
                    final_word = final_word + "Sky, "
                if('Sea' not in original_classes):
                    original_classes = original_classes + ['Sea']
                    final_word = final_word + "Sea, "
                if('Sand' not in original_classes):
                    original_classes = original_classes + ['Sand']
                    final_word = final_word + "Sand "
            elif((objects == 'Sun') and (1 not in count_classes)):
                count_classes.append(1)
                if('Sky' not in original_classes):
                    original_classes = original_classes + ['Sky']
                    final_word = final_word + "Sky, "
                if('Sea' not in original_classes):
                    original_classes = original_classes + ['Sea']
                    final_word = final_word + "Sea, "
                if('Sun' not in original_classes):
                    original_classes = original_classes + ['Sun']
                    final_word = final_word + "Sun "
            elif((objects == 'Tree' or objects == 'grass') and (2 not in count_classes)):
                count_classes.append(2)
                if('Sky' not in original_classes):
                    original_classes = original_classes + ['Sky']
                    final_word = final_word + "Sky, "
                if('Tree' not in original_classes):
                    original_classes = original_classes + ['Tree']
                    final_word = final_word + "Trees, "
                if('grass' not in original_classes):
                    original_classes = original_classes + ['grass']
                    final_word = final_word + "Grass "
            elif(objects == 'mountains' and (3 not in count_classes)):
                count_classes.append(3)
                if('Sky' not in original_classes):
                    original_classes = original_classes + ['Sky']
                    final_word = final_word + "Sky, "
                if('Tree' not in original_classes):
                    original_classes = original_classes + ['mountains']
                    final_word = final_word + "Mountain, "
                if('Tree' not in original_classes):
                    original_classes = original_classes + ['Tree']
                    final_word = final_word + "Trees, "
                if('grass' not in original_classes):
                    original_classes = original_classes + ['grass']
                    final_word = final_word + "Grass "
            elif((objects == 'Sky') and (4 not in count_classes)):
                count_classes.append(4)
                if('Sky' not in original_classes):
                    original_classes = original_classes + ['Sky']
                    final_word = final_word + "Sky "
        return original_classes , final_word
                
    def find_class_possibilities(self , original_classes):
        for objects in original_classes:
            if(objects == "Sea"):
                self.beach_possibility = self.beach_possibility + 1
                self.sunset_possibility = self.sunset_possibility + 1
            if(objects == "Sky"):
                self.beach_possibility = self.beach_possibility + 1
                self.sunset_possibility = self.sunset_possibility + 1
                self.forest_possibility = self.forest_possibility + 0.4
                self.mountain_possibility = self.mountain_possibility + 1
            elif(objects == "Tree"):
                self.forest_possibility = self.forest_possibility + 1
                self.mountain_possibility = self.mountain_possibility + 1
                self.beach_possibility = self.beach_possibility + 0.2
            elif(objects == "Sand"):
                self.beach_possibility = self.beach_possibility + 1
                self.sunset_possibility = self.sunset_possibility + 0.5
            elif(objects == "mountains"):
                self.forest_possibility = self.forest_possibility + 1
                self.mountain_possibility = self.mountain_possibility + 1
                self.beach_possibility = self.beach_possibility + 0.3
                self.sunset_possibility = self.sunset_possibility + 0.1
            elif(objects == "Sun"):
                self.sunset_possibility = self.sunset_possibility + 1
                self.mountain_possibility = self.mountain_possibility + 0.1
                self.beach_possibility = self.beach_possibility + 0.3
            elif(objects == "grass"):
                self.forest_possibility = self.forest_possibility + 1
                self.mountain_possibility = self.mountain_possibility + 1
        
    def display_final_output(self , final_word):
        print(final_word + " (Beach : " + str(round(self.beach_possibility/(self.beach_possibility+self.forest_possibility+self.mountain_possibility+self.sunset_possibility)*100 , 3)) + "%" + " Forest : "  + str(round(self.forest_possibility/(self.beach_possibility+self.forest_possibility+self.mountain_possibility+self.sunset_possibility) *100, 3)) + "%"  +" Mountain : " + str(round(self.mountain_possibility/(self.beach_possibility+self.forest_possibility+self.mountain_possibility+self.sunset_possibility)*100 , 3)) + "%"  + " Sunset : " + str(round(self.sunset_possibility/(self.beach_possibility+self.forest_possibility+self.mountain_possibility+self.sunset_possibility)*100 , 3)) + "%)")   

            
ss = Image_anaysis()
imag,img_for_size = ss.load_image()
img = cv2.fastNlMeansDenoisingColored(imag,None,10,10,7,21)

seg_image , g , labels = ss.segmentation(img)

labels2 = ss.initial_formation(labels , g)

no_of_regions = ss.count_regions(labels2)
print(no_of_regions)
labels2 , reg = ss.customize_threshold_value(labels , g , no_of_regions , 50)
print(reg)

out = ss.merge_regions(img , labels2)
region_rgb_value = ss.extract_color_details(out , labels2)
out = ss.draw_form_image(out , labels2)

region , no_of_pixels_in_region = ss.calculate_pixel_ratio(labels2 , reg)

coordinateVal = ss.extract_region_shape_details(region)

main_dic = ss.merge_feature_set(region_rgb_value , no_of_pixels_in_region , coordinateVal)

feature_vector , removable_keys = ss.create_feature_vector(main_dic)

ori_feature_vector = ss.create_final_feature_vector(feature_vector , removable_keys)

print(ori_feature_vector)

#cv2.imshow("image1" , seg_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#cv2.imshow("image2" , out)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


NN = Neural_Network(Lambda=0.00000001)

training_data = NN.readInputFile("finalDataset.txt")

trainX = training_data[:, [0,1,2,3,4]]
trainY = training_data[:, [5]]


trainY = trainY.reshape(-1,1)
#Scaling
trainX = trainX/np.amax(trainX, axis=0)
trainY = trainY/np.amax(trainY, axis=0) #Max test score is 100

T = trainer2(NN)
T.train2(trainX,trainY)

testingX = np.array((ori_feature_vector))
yHat = NN.forward(testingX)

cc = classification()
classes = cc.identify_objects(yHat)
print(classes)

original_classes , final_word = cc.display_classes(classes)
print(original_classes)

cc.find_class_possibilities(original_classes)

cc.display_final_output(final_word)

        
        