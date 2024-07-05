from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_mobilefacenet_model(input_shape):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_vgg16_model(input_shape):
    base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def data_generator_for_dl(label_file, batch_size=32, target_size=(224, 224)):
    data = pd.read_csv(label_file)
    datagen = ImageDataGenerator(rescale=1./255)
    
    while True:
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_data = data.iloc[start:end]
            batch_images = []
            batch_labels = []
            for _, row in batch_data.iterrows():
                frame_path = row['frame_path']
                label = row['label']
                image = cv2.imread(frame_path)
                if image is not None:
                    image = cv2.resize(image, target_size)
                    image = image / 255.0
                    batch_images.append(image)
                    batch_labels.append(label)
            yield np.array(batch_images), np.array(batch_labels)

input_shape = (224, 224, 3)
mobilefacenet_model = build_mobilefacenet_model(input_shape)
vgg16_model = build_vgg16_model(input_shape)

label_file = '/content/drive/My Drive/CASIA_faceAntisp/labels.csv'
batch_size = 32
epochs = 10

train_gen = data_generator_for_dl(label_file, batch_size=batch_size)

steps_per_epoch = len(pd.read_csv(label_file)) // batch_size

mobilefacenet_model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs)
vgg16_model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs)

mobilefacenet_model.save('mobilefacenet_model.h5')
vgg16_model.save('vgg16_model.h5')
