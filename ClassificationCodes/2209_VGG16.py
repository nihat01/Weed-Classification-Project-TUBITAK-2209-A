import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import VGG16
import seaborn as sn
import pandas as pd

tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_path = r"C:\Users\Zeynep\Desktop\2209_Nihat\DatasetNewIntenet\Train"
test_path = r"C:\Users\Zeynep\Desktop\2209_Nihat\DatasetNewIntenet\Test"
validate_path = r"C:\Users\Zeynep\Desktop\2209_Nihat\DatasetNewIntenet\Valitadtion"

image_width = 224
image_height = 224
learning_rate = 0.000001
batch_size = 16
epoch = 100

def dataset_load_manual(train_path, test_path, validate_path, image_width, image_height):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    validate_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_ds = train_datagen.flow_from_directory(train_path, target_size=(image_width, image_height), color_mode='rgb' ,batch_size=batch_size, class_mode="categorical")
    test_ds = test_datagen.flow_from_directory(test_path, target_size=(image_width, image_height), color_mode='rgb', batch_size=batch_size, class_mode="categorical", shuffle=False)
    validate_ds = validate_datagen.flow_from_directory(validate_path, target_size=(image_width, image_height), color_mode='rgb', batch_size=batch_size, class_mode="categorical", shuffle=False)    
    return train_ds, validate_ds, test_ds

def plot_training(history, model_name):
   
    plt.figure(figsize = (6,4))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('VGG16 Model Doğruluk ve Kayıp Eğrileri')
    plt.ylabel('Doğruluk ve Kayıp')
    plt.xlabel('Çevrim Sayısı')
    plt.legend(['Eğitim Doğruluk Eğrisi', 'Validasyon Doğruluk Eğrisi','Validasyon Kayıp Eğrisi','Eğitim Kayıp Eğrisi'], loc='best')
    plt.ylim(bottom=0, top=1)
    plt.savefig('plots/' + model_name + '_curve.png')
    
class SaveLogs(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filepath, "a") as f:
            f.write(f"Epoch {epoch+1}: {logs}\n")

train_ds, validate_ds, test_ds = dataset_load_manual(train_path, test_path, validate_path, image_width, image_height)
number_of_classes = len(train_ds.class_indices)


#keras kütüphanesinden çağırılan hazır model. weights: imagenet gibi ön eğitimli ağırlıklar çağırılabilir.
#include_top değişkeni true yapıldığı zaman modelin kendi fully connected layer katmanları çağırılıyor.
base_model = VGG16(weights= "imagenet", include_top=False, input_shape=(image_width, image_height, 3))

for layer in base_model.layers:
    layer.trainable = True
    
#base modelin fully conn layerları yerine kendi eklediğimiz çıktı katmanlarını ekliyoruz.
model_output = base_model.output
model_output = tf.keras.layers.GlobalAveragePooling2D()(model_output)
model_output = tf.keras.layers.Dropout(0.3)(model_output)
model_output = tf.keras.layers.Dense(1024, activation='relu')(model_output)
model_output = tf.keras.layers.Dropout(0.3)(model_output)
model_output = tf.keras.layers.Dense(512, activation='relu')(model_output)

prediction = tf.keras.layers.Dense(number_of_classes, activation='softmax')(model_output)

# input kısmına keras içerisinden çağırılan hazır model inputlarını, output kısmına ise kendi eklediğimiz katmanları ekliyoruz.
model_vgg_train = tf.keras.models.Model(inputs=base_model.inputs, outputs=prediction)
 
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=1e-05)
model_vgg_train.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

logfile = "plots/" + "VGG16" + "_training_output.txt"

start_train = datetime.now()
history = model_vgg_train.fit(train_ds, validation_data = validate_ds, epochs=epoch,shuffle=True,callbacks=[SaveLogs(logfile)])                                                                                            
end_train = datetime.now()

plot_training(history,'VGG16')

print("Training duration : {}".format(end_train-start_train))

start_test = datetime.now()
result_final = model_vgg_train.evaluate(test_ds)
end_test = datetime.now()
print("Test duration : {}".format(end_test-start_test))

print("test loss : ", result_final[0])
print("test accuracy : ", result_final[1])

with open("plots/" + "VGG16" + "_training_output.txt", "a") as f:
    f.write("Training duration : {}\n".format(end_train - start_train))
    f.write("Test duration : {}\n".format(end_test - start_test))
    f.write("test loss : {}\n".format(result_final[0]))
    f.write("test accuracy : {}\n".format(result_final[1]))

# confusion matrix
sn.set(font_scale=1.4)
predict = model_vgg_train.predict(test_ds)
predict = np.argmax(predict, axis=1)
cm = confusion_matrix(test_ds.classes, predict)

a = ["Amaranthus", "Chenopodium","Solanum","Convolvulus","Salsola" ]  
    
df_cm = pd.DataFrame(cm, index = [i for i in a],
                  columns = [i for i in a])
plt.figure(figsize = (14,10))

plt.title("VGG16 Karışıklık Matrisi", fontsize=20)

s = sn.heatmap(df_cm, annot=True ,cmap="YlGn",annot_kws={"size":20})

s.set(xlabel='Tahmin Edilen Sınıflar', ylabel='Gerçek Sınıflar')
cm_report = classification_report(test_ds.classes, predict, output_dict=True, digits=7)
df_cm_report = pd.DataFrame(cm_report).transpose()
class_names = [i for i in a]
class_names.append("Accuracy")
class_names.append("Macro Avg")
class_names.append("Weighted Avg")
df_cm_report.insert(0, "class_names", class_names, True)

print(df_cm_report)

with open("plots/" + "VGG16" + "_training_output.txt", "a") as f:
    f.write(df_cm_report.to_string())

plt.savefig('plots/' + 'VGG16' + '_matrix.png')



