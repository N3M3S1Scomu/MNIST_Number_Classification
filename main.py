


# DATASET OKUMA
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

result=x_train.shape # (60000, 28, 28)
result=len(x_train) # goruntu sayısı-> 60000

# yapay sinir ağı mimarisi
from keras.models import Sequential
from keras.layers import Dense

model=Sequential() # modeli olusturduk

model.add(Dense(512,activation="relu",input_shape=(28*28,))) # düz bir sinir agı ekledik
# 512 = eleman/sinir hücresi sayısı

model.add(Dense(10,activation="softmax")) # çıkış katmanı

# YSA modelinin derlenmesi
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"] # başarı metrigi = accuracy
              )

# train,test verisi hazırlama
x_train=x_train.reshape((60000,(28*28)))
x_train=x_train.astype("float32")/255

x_test=x_test.reshape((10000,(28*28)))
x_test=x_test.astype("float32")/255

from keras.utils import to_categorical

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
"""
# YSA MODELİNİN EĞİTİLMESİ
model.fit(x_train,y_train,epochs=5,batch_size=128)
model.save("model.h5")
"""

# tahmin
from keras.models import load_model
import numpy as np

model=load_model("model.h5")

test_loss,test_acc = model.evaluate(x_test,y_test) # 0.07  0.97

print(test_loss,test_acc)

preds=np.argmax(model.predict(x_test), axis=-1)
y_test=np.argmax(y_test,axis=-1)

print(preds[0:5],y_test[:5])

"""
If your model performs multi-class classification (e.g. if it uses a softmax last-layer activation) use: 

np.argmax(model.predict(x), axis=-1)

If your model performs binary classification (e.g. if it uses a sigmoid last-layer activation) use:

(model.predict(X_test) > 0.5).astype("int32")
"""




