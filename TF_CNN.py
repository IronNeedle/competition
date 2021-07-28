from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.optimizers import SGD

base_model = MobileNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(100, activation='relu')(x)
x = Dense(25, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


base_dir = 'C:/tmp/TRAIN/'
val_dir = 'C:/tmp/VALIDATION/'
test_dir = 'C:/tmp/TEST/'
batch_size = 20

test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(
     rescale=1./255,
     rotation_range=40,
     width_shift_range=0.20,
     height_shift_range=0.20,
     shear_range=0.2,
     zoom_range=0.2,
     horizontal_flip=True,
     fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=50
)

for layer in model.layers[:80]:
    layer.trainable = False
for layer in model.layers[80:]:
    layer.trainable = True

model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=1500,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[EarlyStopping(monitor='accuracy', patience=100)]
)


model_json = model.to_json()
with open("MobileNetV2_1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("MobileNetV2_1.h5")

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)



hist_df = pd.DataFrame(history.history)
hist_df.to_csv('TF_MobileNetV2_1_HISTORY.csv')

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='training_loss')
plt.plot(epochs, val_loss, 'b', label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

loss = history.history['accuracy']
val_loss = history.history['val_accuracy']

plt.plot(epochs, loss, 'bo', label='acc')
plt.plot(epochs, val_loss, 'b', label='val_acc')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

test_loss, test_acc, _ = model.evaluate(test_generator, steps=80)
print('test acc:', test_acc)
