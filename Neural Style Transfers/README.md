NEURAL STYLE TRANSFERS  

In Neural Style Transfers the art is generated using the STYLES like brush strokes, textures,etc of one image and drawing them on the CONTENT of another image.  
This project uses pretrained VGG16 model with imagenet weights for the art geneartion.  
The total loss in this case would be the sum of content loss and style loss and we obviously try to minimize it.  

CONTENT IMAGE..  
![](images/eiffel.jpg)  
STYLE IMAGE..  
![](images/thescream.jpg)  
OUTPUT IMAGE..  
![](outputs/thescream_onto_eiffel_at_iteration_9.png)
