from keras.layers import Dense, Dropout, Input, ReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import mnist #dataset
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#data load edildi. 
x_train = (x_train.astype(np.float32)-127.5)/127.5

print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
print(x_train.shape) #(28,28) =784

#%% 
#plt.imshow(x_test[12])

#%% create generator
def create_generator():
    
    generator = Sequential()
    generator.add(Dense(units = 512, input_dim = 100))
    generator.add(ReLU())
    
    generator.add(Dense(units = 512))
    generator.add(ReLU())
    
    generator.add(Dense(units = 1024))
    generator.add(ReLU())
    
    generator.add(Dense(units = 784, activation = "tanh"))
    
    generator.compile(loss = "binary_crossentropy", #fake yada fake olmayan resim bu sebeple binary_crossentropy
                      optimizer = Adam(lr = 0.0001, beta_1 = 0.5))
    return generator

g = create_generator()
g.summary()   
#generater oluşturuldu. 
    
#%% dsicriminator

def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(units=1024,input_dim = 784))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))
    
    discriminator.add(Dense(units=512))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))
    
    discriminator.add(Dense(units=256))
    discriminator.add(ReLU())
    
    discriminator.add(Dense(units=1, activation = "sigmoid"))
    
    discriminator.compile(loss = "binary_crossentropy",
                          optimizer= Adam(lr = 0.0001, beta_1=0.5))
    return discriminator

d = create_discriminator()
d.summary()
    
#discriminator oluşturuldu. Bunun için Dense metodu ile layerlar eklendi,aktivasyon fonksiyonları verildi.

#%% gans
def create_gan(discriminator, generator):
    discriminator.trainable = False #discriminator train edilmesi false
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs = gan_input, outputs = gan_output) #modeli oluşturduk
    gan.compile(loss = "binary_crossentropy", optimizer="adam")
    return gan
    
gan = create_gan(d,g)
gan.summary()

#GAN oluşturuldu. Input değerleri generatora girdi olarak verildi. 
#Çıktısı discriminatora verildi.

# %% train

epochs = 50
batch_size = 256

for e in range(epochs):
    for _ in range(batch_size):
        
        noise = np.random.normal(0,1, [batch_size,100]) #0ve 1 arası noise 
        
        generated_images = g.predict(noise) #generater(g) noise alacak ve generated_images leri verecek.
        
        image_batch = x_train[np.random.randint(low = 0, high = x_train.shape[0],size = batch_size)] #randint integer değerler döndürür.
                                                                                                   #256 adet gerçek imageler alınacak
        
        x = np.concatenate([image_batch, generated_images])
        
        y_dis = np.zeros(batch_size*2) #512 tane sıfır
        y_dis[:batch_size] = 1 #y_dis teki ilk 256 değer 1 denildi çünkü gerçek imageler
        
        d.trainable = True
        d.train_on_batch(x,y_dis)

        noise = np.random.normal(0,1,[batch_size,100]) #generator eğitmek için
        
        y_gen = np.ones(batch_size) #discriminator kandırmak için 1  denildi
        
        d.trainable = False
        
        gan.train_on_batch(noise, y_gen)
    print("epochs: ",e)



#%% save model
g.save_weights('gans_model.h5') 


#%% visualize
noise= np.random.normal(loc=0, scale=1, size=[100, 100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
plt.imshow(generated_images[66], interpolation='nearest')
plt.axis('off')
plt.show()

























