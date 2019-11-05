import face_recognition

# Load the jpg files into numpy arrays

LiQin_image = face_recognition.load_image_file("../input/LiQin.jpg")
SunYi_image = face_recognition.load_image_file("../input/SunYi.jpg")
unknown_image = face_recognition.load_image_file("../input/Unknown.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    LiQin_face_encoding = face_recognition.face_encodings(LiQin_image)[0]
    SunYi_face_encoding = face_recognition.face_encodings(SunYi_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    LiQin_face_encoding,
    SunYi_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding, tolerance=0.45)

print("Is the unknown face a picture of LiQin? {}".format(results[0]))
print("Is the unknown face a picture of SunYi? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))
