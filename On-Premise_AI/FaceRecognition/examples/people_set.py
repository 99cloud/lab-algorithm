# Python 3.3 +

import face_recognition


class people:
    # 定义基本属性
    name = ''
    image = ''
    face_encoding = ''

    # 定义构造方法
    def __init__(self, n, path):
        self.name = n
        self.image = face_recognition.load_image_file(path + n + ".jpg")
        self.face_encoding = face_recognition.face_encodings(self.image)[0]


def main():
    path = '../input/'
    name = 'heyujia'
    s = people(name, path)
    print('name:', s.name, '\nencoding:', s.face_encoding)


if __name__ == '__main__':
    main()
