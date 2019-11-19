import cv2
import os

camera = cv2.VideoCapture(0)
for i in range(0,2):
    return_value, image = camera.read()
    cv2.imwrite('C:/Users/Chiranthana/Pictures/'+'opencv'+str(i)+'.jpg', image)
del(camera)



from keras.models import load_model
import numpy as np
from keras.preprocessing import image
#train_image_gen.class_indices
model=load_model('C:/Users/Chiranthana/Pictures/so_si.h5')
image_file = 'C:/Users/Chiranthana/Pictures/'+'opencv'+str(i)+'.jpg'


img = image.load_img(image_file, target_size=(150, 150))

img = image.img_to_array(img)

img = np.expand_dims(img, axis=0)
img = img/255

prediction_prob = model.predict(img)
if prediction_prob<0.5:
    print(prediction_prob)
    print ("shirt")
    os.system('python t-shirt.py ')
    
else:
    print(prediction_prob)

    print("short")
    os.system('python shot.py ')
