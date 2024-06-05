import glob
import numpy as np 
import onnxruntime as rt
from onnxruntime.datasets import get_example
import json
import cv2
import Detect_Image_TooLight_TooDarK


def detect_image_too_dark(model,image_file):


# Load a very simple model (e.g., sigmoid.onnx)
    
    sess = rt.InferenceSession(model)

    # Get input details
    input_name = sess.get_inputs()[0].name

    # Get output details
    output_name = sess.get_outputs()[0].name

    # image = glob.glob("D:/Detect_too_light_MLP/test/dark"+"/*.jpg")
    
    image=cv2.imread(image_file)
    image=Detect_Image_TooLight_TooDarK.give_value_to_image_with_mxn(image,8,8,"dark").astype(np.float32)
    image=image.reshape(-1, 8,8,1)
    res=sess.run([output_name],{input_name:image})
    
    if np.argmax(res)==1:
        return True
    return False
def detect_image_too_light(model,image_file):
    
    sess = rt.InferenceSession(model)

    # Get input details
    input_name = sess.get_inputs()[0].name

    # Get output details
    output_name = sess.get_outputs()[0].name

    # image = glob.glob("D:/Detect_too_light_MLP/test/dark"+"/*.jpg")
    
   

    image=cv2.imread(image_file)
    image=Detect_Image_TooLight_TooDarK.give_value_to_image_with_mxn(image,8,8,"light").astype(np.float32)
    image=image.reshape(-1, 8,8,1)
    res=sess.run([output_name],{input_name:image})
    
    if np.argmax(res)==1:
        return True
    return False
def blur_image(threshold,image_file):
    image=cv2.imread(image_file)
    if Detect_Image_TooLight_TooDarK.check_blur_image(image,threshold):
        return True
    return False
    
    
if __name__ == "__main__":
    model_dark="model/detect_too_dark_5_6.onnx"
    model_light="model/detect_too_light_4_6.onnx"
    file_image_list=glob.glob("tmp/image\*.jpg")
    image_list_error=[]
    json_file_detected=[]
    for file_image in file_image_list:
        json_file_detected.append({"image_name":file_image,
                                   "blur":blur_image(50,file_image),
                                   "too_light":detect_image_too_light(model_light,file_image),
                                   "too_dark":detect_image_too_dark(model_dark,file_image)})
    with open("result.json","w") as f:
        json.dump(json_file_detected, f, ensure_ascii=True)
    print("successfully detected")