#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow==2.4.1


# In[2]:


pip install tensorflow-gpu==2.4.1


# In[3]:


pip install pixellib


# In[4]:


pip install opencv-python


# In[5]:


pip install numpy --upgrade


# In[6]:


pip install tensorfkow --upgrade


# In[7]:


pip install imgaug


# In[8]:


pip install pixellib==0.4.8


# In[9]:


pip install pixellib --upgrade


# In[10]:


pip install numpy --upgrade


# In[11]:


import pixellib
from pixellib.instance import instance_segmentation
import cv2


# In[12]:


segmentation_model = instance_segmentation()
segmentation_model.load_model(r'C:\Users\panda\Downloads\mask_rcnn_coco.h5')


# In[ ]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Apply instance segmentation
    res = segmentation_model.segmentFrame(frame, show_bboxes=True)
    image = res[1]
    
    cv2.imshow('Instance Segmentation', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'segmentation_model.segmentFrame')


# In[ ]:


# press Q for clos the camera


# In[ ]:





# In[ ]:





# In[ ]:




