** Model Structure

I used a model that was based on the model described in the Nvidia paper. I modified the model in the paper by following each convolutional layer with max pooling and dropout. I also removed one fully connected layer to decrease parameter size. The total number of parameters was 136,443.

This is the output from model.summary():
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 62, 196, 24)   1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 62, 196, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 31, 98, 24)    0           elu_1[0][0]                      
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 31, 98, 24)    0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 27, 94, 36)    21636       dropout_1[0][0]                  
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 27, 94, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 13, 47, 36)    0           elu_2[0][0]                      
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 13, 47, 36)    0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 11, 45, 48)    15600       dropout_2[0][0]                  
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 11, 45, 48)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 5, 22, 48)     0           elu_3[0][0]                      
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 5, 22, 48)     0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       dropout_3[0][0]                  
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 3, 20, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 1, 10, 64)     0           elu_4[0][0]                      
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 1, 10, 64)     0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 640)           0           dropout_4[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           64100       flatten_1[0][0]                  
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 100)           0           elu_5[0][0]                      
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_5[0][0]                  
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         elu_6[0][0]                      
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 10)            0           elu_7[0][0]                      
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_6[0][0]                  
====================================================================================================
Total params: 136,443
Trainable params: 136,443
Non-trainable params: 0

** 
Training Approach

While I experimented with gathering different datasets using both a keyboard and joystick, I was unable to get my model to perform well on the second track with any of them. For my final submission I am using the Udacity dataset that was provided. I created a generator function that randomly selected driving log entries and performed some simple augmentation. Left and Right camera images were used by offsetting the center steering angle with random values in the range of .15 - .3. I resized the images to 200x66 which is what was used in the original Nvidia paper. I randomnly adjusted the brightness of the images by converting to HSV and randmonly scaling the V channel and then converting back to RGB. In addition I performed random horizontal translation and flipping of center images.

Using a .001 learning rate and training on 10 epochs I also tried to offset the bias of the straight steering angle log entries by discarding straight (< .1) steering angle images using a decaying probability parameter in between each epoch. It should be noted that this model with this particular training methodology does not make it past the first turn on track 2. While I was able to get a different training data approach to perform better on track 2 it did not make it past the 3rd turn and I am submitting this particular methodology because it is simpler and more straightforward to describe.

UPDATED:

I added a validation data generator which was suggested after my first project review. The validation data generator used a seperate dataset that I collected on my own. I also added a ModelCheckpoint callback that checked the validion error and terminated training when the validation error was no longer improving on subsequent epochs. 


