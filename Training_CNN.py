import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout  
from keras.layers import Conv2D,MaxPooling2D 
from keras.callbacks import ModelCheckpoint  
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt 

data = np.load('data.npy')
target = np.load('target.npy')
      
model = Sequential()

#first convolution layer followed by relu and maxpooling
model.add(Conv2D(200,(3,3),input_shape = data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2))) 

#second convolution layer followed by relu and maxpooling
model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2))) 

#Flatten Layer
model.add(Flatten())
model.add(Dropout(0.5))
    
#Dense Layer 50
model.add(Dense(50,activation="relu"))

#Dense Layer 2
model.add(Dense(2,activation="softmax"))

model.compile(loss = "categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

train_data,test_data,train_target,test_target = train_test_split(data,target,test_size = 0.1)

checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history = model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)

plt.plot(history.history['loss'],'r',label='Training Loss')
plt.plot(history.history['val_loss'],'r',label='Validation Loss')    
plt.xlabel('# epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(model.evaluate(test_data,test_target))

    
    