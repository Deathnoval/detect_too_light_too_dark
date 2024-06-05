import json
import cv2

with open ("result.json", "r") as result:
    data_list=json.load(result)
   
for data in data_list:
    if data["too_light"] ==True:
        image = cv2.imread(data["image_name"])
        image=cv2.resize(image,(400,400 ))
        cv2.imshow(data["image_name"],image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()