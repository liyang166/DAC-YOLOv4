import os
#import cv2

def rename_image(path):
    for root, dirs, files in sorted(os.walk(path)):
        for filename in files:
            nameList=filename.split('.')
            #print nameList
            if int(nameList[0]) > 4377 and nameList[1]=='jpg':
                filenew=nameList[0]+'.png'
                print(filenew)
                os.rename(os.path.join(root,filename),os.path.join(root,filenew))

if __name__ == "__main__":
    rename_image("/home/lab346_admin/data/dataset/insect/image")