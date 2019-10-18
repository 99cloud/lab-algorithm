from PIL import Image, ImageDraw

# Load pictures
pil_image = Image.open('img/Blackpink_2.jpeg')

# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Input Parameter
face_num = 4
face_rect = [172, 145, 39, 55, 472, 160, 38, 55, 363, 200, 44, 56, 267, 171, 43, 56]
ages = [25, 24, 24, 29]
expressions = [1, 0, 1, 1]

# Swift Parameter
smile_faces = ['smile' if x == 1 else 'normal' for x in expressions]
face_rect = zip(face_rect[0::face_num], face_rect[1::face_num], face_rect[2::face_num], face_rect[3::face_num])

# Loop through each face
for (left, top, width, height), age, smile_face in zip(list(face_rect), ages, smile_faces):

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (left + width, top + height)), outline=(255, 0, 0), width=2)

    # Draw a label with age and smile below the face
    text_width, text_height = draw.textsize(smile_face)
    draw.rectangle(((left, top + height), (left + width, top + height + text_height + 20)), fill=(255, 0, 0), outline=(255, 0, 0))
    draw.text((left + 2, top + height + text_height - 5), smile_face + '\nage:' + str(age), fill=(255, 255, 255, 255))

# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()

# You can also save a copy of the new image to disk if you want by uncommenting this line
pil_image.save("../img/Blackpink_2_attribute.jpg")