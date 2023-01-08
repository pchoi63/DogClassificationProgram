#Younghoon Choi
#Professor Douglass
#COM 307
#11/29/2022

#####Import Libs#####
from bing_image_downloader import downloader
import os

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import svm
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

#####Global Stuff######

#List
target = []
images = []
imgFlattened = []

#Used for downloading images from bing.com
#I ended up just creating my own dataSet since bing sucks
def downloadImages():
    downloader.download("Maltese", limit = 300, output_dir = 'images', adult_filter_off = True)
    downloader.download("Chow Chow", limit = 300, output_dir = 'images', adult_filter_off = True)
    downloader.download("Golden Retriever", limit = 300, output_dir = 'images', adult_filter_off = True)

def URLmode():
    url = input('Enter your url')
    img = imread(url)
    img_resized = resize(img, (150,150,3))
    imgFlattened.append(img_resized.flatten())
    flat_data = np.array(imgFlattened)
    plt.imshow(img_resized)
    
    y_out = model.predict(flat_data)
    y_out = Categories[y_out[0]]
    print(f'This dog is {y_out}')
    plt.show()


#Process the images from the folder
#1. Read the file
#2. Resize it
#3. Flatten it
#4. Train it
def preprocessing(category):
    #List
    imgFlattened = []
    target = []
    images = []

    #For each index in Categories List, breeds of dogs.
    for category in Categories:
        #Get the index of current category
        class_num = Categories.index(category)
        print(category)
        #Direct to the file currently in, hardcoded below 
        #Checks through all the folders in images folder
        path = os.path.join(dataDir, category)

        #Read each image in the path
        for img in os.listdir(path):
            #Read the image
            imgArray = imread(os.path.join(path, img))

            #Resize to match with the testing images
            imgResized = resize(imgArray,(150,150,3))

            #Flatten the image
            imgFlattened.append(imgResized.flatten())

            #Size of the data
            target.append(class_num)
            
    #Use numpy
    imgFlattened = np.array(imgFlattened)

    #size of the data
    target = np.array(target)

    #put the 
    trainData(imgFlattened, target)

#This funtion reads test Images in the testIm file
#Works the same way as preprocessing function above without the training part
def testImages(testCategories, model):
    #used for total numbers
    count = 0
    index = 0
    score = 0
    for testCategories in testCategories:
        path = os.path.join(testdataDir, testCategories)
        #used for each category breed
        currCount = 0
        currScore = 0
        currInd = 0
        
        for img in os.listdir(path):
            #Read, resize and flat then predict
            imgArray = imread(os.path.join(path, img))
            img_resized = resize(imgArray, (150,150,3))
            imgFlattened.append(img_resized.flatten())
            flat_data = np.array(imgFlattened)
            y_out = model.predict(flat_data)
            y_out = Categories[y_out[count]]
            print(f'Predicted output for \n{img} = {y_out}\n')
            #plt.imshow(img_resized)
            #plt.show()
            currCount = currCount + 1
            currInd = currInd + 1

            #counting score for both total and category
            if(y_out == Categories[index]):
                score = score + 1
                currScore = currScore + 1
            
            count = count+1
        
        #Print the category accuracy
        print("Accuracy for " + Categories[index]+ ": " +str(currScore/currInd * 100) + "%\n")
        index = index + 1

    return score, count

#flatData is flatten images from preprocessing
#Target is the size of them
def trainData(flatData, target):
    #Split the Data into train and test at random state
    #Not really used but here to show the method in other ways
    x_train, x_test, y_train, y_test = train_test_split(flatData, target, test_size=0.1, random_state=109)

    #Parameter for GridSearch
    param_grid =[
        {'C':[1,10,100,1000], 'kernel':['linear']},    
        {'C':[1,10,100,1000], 'gamma':[0.001, 0.0001],'kernel':['rbf']}
    ]

    #Set up support vector machine
    svc = svm.SVC(probability= True)

    #GridSearch from sklearn
    clf = GridSearchCV(svc, param_grid)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    #score = accuracy_score(y_pred, y_test)
    #matrixScore = confusion_matrix(y_pred, y_test)
    pickle.dump(clf,open('img_model.p', 'wb'))



#####_Main_#####
if __name__ == "__main__":
    print("This program classifies dog breed!")

    #Used for folder names
    Categories = ['Beagle', 'Bulldog', 'Golden_retriever', 'Greyhound', 'Pomeranian']   #In images folder
    testCategories = ['testBeagle','testBulldog',  'testGolden', 'testGreyhound', 'testPomeranian']     #In testIm folder

    #Get current directory
    baseDir = os.getcwd()

    #Get path to training images 
    dataDir = baseDir + '/images'

    #Get path to test images
    testdataDir = baseDir + '/testIm'
    
    #downloadImages()
    #preprocessing(Categories)

    #Create model and save it to a file
    model = pickle.load(open('img_model.p', 'rb'))
    
    rightIm, totalIm = testImages(testCategories, model)

    rightIm, totalIm = neuralTestImages(testCategories, model)
    print("Total Accuracy: " + str((rightIm)/totalIm * 100)+ "%")
    print(str(rightIm) + " out of " + str(totalIm)+ " were predicted correctly")
    
    URLmode()