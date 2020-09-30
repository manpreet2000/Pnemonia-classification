import tensorflow as tf
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("path", help="path of image",
                    type=str)
args = parser.parse_args()

base_model=tf.keras.applications.densenet.DenseNet121(include_top=False)
x=base_model.output
x=tf.keras.layers.GlobalAveragePooling2D()(x)

x=tf.keras.layers.Dropout(0.5)(x)

pred=tf.keras.layers.Dense(1,activation='sigmoid')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=pred)

model.load_weights("/home/manpreet/codes/medical/pnumonia/weights/my_model_weights.h5")
print("")
print("Loading Image")
img=tf.keras.preprocessing.image.load_img(args.path,target_size=(128,128))
img=tf.keras.preprocessing.image.img_to_array(img)
img=np.expand_dims(img,axis=0)
print("Predicting ")
result=model.predict(img)
if result[0][0]<0.9999456:
    result="Normal"
else:
    result="Pnemonia"

print("*"*50)
print("Result is...")
print(result)