import face_recognition
import os
import cv2 as cv

known_faces_dir="knownfaces"
unknown_faces_dir="unknownfaces"

TOLERANCE=0.6
FRAME_THICKNESS=3
FONT_THICKNESS=2
MODEL='cnn'

def identifyfaces():


    known_faces=[]
    known_names=[]

    for name in os.listdir(known_faces_dir):
        if name!='.DS_Store':
            for filename in os.listdir(f"{known_faces_dir}/{name}"):
                if '.jpg' in filename.lower() or '.jpeg' in filename.lower() or '.png' in filename.lower():
                    image=face_recognition.load_image_file(f"{known_faces_dir}/{name}/{filename}")
                    encodings=face_recognition.face_encodings(image)[0]

                    known_faces.append(encodings)
                    known_names.append(name)
                else:
                    continue
        else:
            continue
    
    for filename in os.listdir(unknown_faces_dir):
        if '.jpg' in filename.lower() or '.jpeg' in filename.lower() or '.png' in filename.lower():
            image=face_recognition.load_image_file(f"{unknown_faces_dir}/{filename}")
            locations=face_recognition.face_locations(image,model=MODEL)
            encodings=face_recognition.face_encodings(image, locations)


            for face_encoding,face_location in zip(encodings,locations):
                results=face_recognition.compare_faces(known_faces,face_encoding,TOLERANCE)
                match=None
                if True in results:
                    match=known_names[results.index(True)]
                    print(f"Match found: {match}")

                    p1=(face_location[3],face_location[0])
                    p2=(face_location[1],face_location[2])
                    color=[255,0,0]
                    cv.rectangle(image, p1,p2,color,FRAME_THICKNESS)

                    p1=(face_location[3],face_location[2])
                    p2=(face_location[1],face_location[2]+22)
                    color=[255,0,0]
                    cv.rectangle(image, p1,p2,color,FRAME_THICKNESS)
                    cv.putText(image, match, (face_location[3]+10, face_location[2]+15),cv.FONT_HERSHEY_SIMPLEX,0.5,[255,255,255],FONT_THICKNESS)

            image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
            cv.imshow(filename, image)
            cv.waitKey(10000000)
        else:
            continue
        


def findfaces():
    
    for filename in os.listdir(unknown_faces_dir):
        if '.jpg' in filename.lower() or '.jpeg' in filename.lower() or '.png' in filename.lower():
            image=face_recognition.load_image_file(f"{unknown_faces_dir}/{filename}")
            face_locations = face_recognition.face_locations(image, model=MODEL)
        

            for face_location in face_locations:
                p1=(face_location[3],face_location[0])
                p2=(face_location[1],face_location[2])
                color=[255,0,0]
                cv.rectangle(image, p1,p2,color,FRAME_THICKNESS)
            
            image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
            cv.imshow(filename, image)
            cv.waitKey(1000000)
        else:
            continue
        

#Display Menu
ch=int(input("1) Detect Faces in images\n2) Identify Faces in images\nEnter Choice: "))
if ch==1:
    findfaces()
elif ch==2:
    identifyfaces()
else:
    print("Wrong Choice")
