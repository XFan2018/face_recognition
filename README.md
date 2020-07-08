# face_recognition 
1. face dataset was created with webcam and face enrollment.
2. face detection was used to extract the face features from the face dataset.
3. face detection model is pretrained res10_300x300_ssd_iter_140000.caffemodel from https://www.pyimagesearch.com.
4. extracted face features are embedded by faceNet impplemeted with pytorch to generate 128d embedddings.
5. embeddings are to be used for face recognition
