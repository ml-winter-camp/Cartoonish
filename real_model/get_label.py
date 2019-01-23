import os

def to_int (str):
    if str == "1":
        n = 1
    else:
        n = 0
    return n

path = os.getcwd()

label = open('./list_attr_celeba.txt', 'r')
f1 = open('Mustache_train_label.txt', 'w')
f2 = open('Mustache_val_label.txt', 'w')
f3 = open('Mustache_test_label.txt', 'w')

count = 1
for line in label:
    x = line.strip().split(' ');
    if len(x) < 40:
        print("length error" + str(len(x)))
        continue
    if (not (x[1] == "1")) and (not(x[1] == "-1")):
        continue
    try:
        has_glasses = to_int(x[24])
    except Exception as e:
        continue
    x[0] = x[0].replace('jpg', 'png')
    if count <= 162770:
        f1.write(path + "/img_align_celeba_png/" + x[0] + " " + str(has_glasses) + '\n')
    elif count <= 182637:
        f2.write(path + "/img_align_celeba_png/" + x[0] + " " + str(has_glasses) + '\n')
    else:
        f3.write(path + "/img_align_celeba_png/" + x[0] + " " + str(has_glasses) + '\n')
    count += 1
label.close()
f1.close()
f2.close()
f2.close()
