from PIL import Image, ImageDraw
import numpy as np
import os
import json
import pprint


def check(data_1):

    #img = Image.open("/Users/drvish/Desktop/TuftsDentalData/Radiographs/" + data_1["External ID"].split(".")[0] + ".JPG")
    img = Image.open("/Users/drvish/Desktop/TuftsDentalData/Radiographs/" + data_1["External ID"])
    draw = ImageDraw.Draw(img)


    for item in data_1["Label"]["objects"]:
        text = item["title"]
        x, y, w, h  = item["bounding box"]
        draw.rectangle([x, y, x+w, y+h], outline='green', width=2)
        draw.text([x, y-10], text)

    img.show()
    img.save('result.jpg')



def trial():

    with open("/Users/drvish/Desktop/TuftsDentalData/Segmentation/teeth_polygon.json", "r") as f:
        file_contents = json.load(f)
    
    print(len(file_contents))

    data_1 = {}

    for i in range(len(file_contents)):
        d = file_contents[i]
        
        if d["External ID"] == "1.JPG":
        #if file_contents[i]["Label"]["objects"]["External ID"] == "1.JPG":
            check(d)

   

import cv2


def check2(data_1):

    #img = Image.open("/Users/drvish/Desktop/TuftsDentalData/Radiographs/" + data_1["External ID"].split(".")[0] + ".JPG")
    img = Image.open("/Users/drvish/Desktop/TuftsDentalData/Radiographs/" + data_1["External ID"])
    path = "/Users/drvish/Desktop/TuftsDentalData/Radiographs/" + data_1["External ID"]
    #draw = ImageDraw.Draw(img)
    image = cv2.imread(path)
    file_name = "roi_" + data_1["External ID"]


    for o in data_1["Label"]["objects"]:
        folder_path = "/Users/drvish/Desktop/roi"
        if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        for polygons in o["polygons"]:
            file_name = os.path.join(folder_path, "roi_" + data_1["External ID"])
            #if len(o["polygons"]) > 1:
            #    image = cv2.imread("/Users/drvish/Desktop/roi/" + data_1["External ID"])
            pts = np.array(polygons, np.int32)
            #draw.rectangle([x, y, x+w, y+h], outline='green', width=2)
            #draw.text([x, y-10], text)
            # draw.polygon(xy, fill ="#eeeeff", outline ="blue") 
            
            print(pts)

            pts = pts.reshape((-1, 1, 2))

            isClosed = True
            color = (255, 0, 0)

            thickness = 2

            image = cv2.polylines(image, [pts], isClosed, color, thickness)
            cv2.imwrite(file_name, image)
            
        #cv2.destroyAllWindows()
            #cv2.imwrite(file_name, image)
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)
            # file_name = os.path.join(folder_path, "roi_" + data_1["External ID"])
            # cv2.imwrite(file_name, image)
            # cv2.imwrite(file_name, image)
        


def trial2():
    with open("/Users/drvish/Desktop/TuftsDentalData/Expert/expert.json", "r") as f:
        file_contents = json.load(f)

    for i in range(len(file_contents)):
        d = file_contents[i]
        print(d["External ID"])
        if d["Label"]["objects"][0]["polygons"] != "none":                
            check2(d)
    
    
    #check2(c)

trial2()
