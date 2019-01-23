import os

def to_int (str):
    if str == "1":
        n = 1
    else:
        n = 0
    return n

path = os.getcwd()

label = open('./list_attr_celeba.txt', 'r')
f1 = open('hair_color_train_label.txt', 'w')
f2 = open('hair_color_val_label.txt', 'w')
f3 = open('hair_color_test_label.txt', 'w')

count = 1
for line in label:
    x = line.strip().split(' ');
    if len(x) < 40:
        print("length error" + str(len(x)))
        continue
    if (not (x[1] == "1")) and (not(x[1] == "-1")):
        continue
    try:
        is_black = to_int(x[10])
        is_blond = to_int(x[11])
        is_brown = to_int(x[13])
    except Exception as e:
        continue
    if is_black == 1:
        is_blond = 0
        is_brown = 0
    if is_blond == 1:
        is_brown = 0
    is_other = 0
    if (is_black == 0 and is_blond == 0 and is_brown == 0):
        is_other = 1
    if is_black == 1:
        cur_label = 0
    elif is_blond == 1:
        cur_label = 1
    elif is_brown == 1:
        cur_label = 2
    else:
        cur_label = 3
    x[0] = x[0].replace('jpg', 'png')
    if count <= 162770:
        f1.write(path + "/img_align_celeba_png/" + x[0] + " " + str(cur_label) + '\n')
    elif count <= 182637:
        f2.write(path + "/img_align_celeba_png/" + x[0] + " " + str(cur_label) + '\n')
    else:
        f3.write(path + "/img_align_celeba_png/" + x[0] + " " + str(cur_label) + '\n')
    count += 1
label.close()
f1.close()
f2.close()
f2.close()
