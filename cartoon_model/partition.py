f_label = open('label_glasses.txt','r')
f_part1 = open('glasses_train.txt','w')
f_part2 = open('glasses_test.txt','w')
i = 1
for row in f_label:
	# print(row)
	# f_part1.writelines(row)
	if i <= 60000:
		f_part1.writelines(row)
	else:
		f_part2.writelines(row)
	i = i + 1

f_part1.close()
f_part2.close()
f_label.close()

f_label2 = open('label_haircolor.txt','r')
f_part1 = open('haircolor_train.txt','w')
f_part2 = open('haircolor_test.txt','w')
i = 1
for row in f_label2:
	# print(row)
	# f_part1.writelines(row)
	if i <= 60000:
		f_part1.writelines(row)
	else:
		f_part2.writelines(row)
	i = i + 1

f_part1.close()
f_part2.close()
f_label2.close()

f_label3 = open('label_bald.txt','r')
f_part1 = open('bald_train.txt','w')
f_part2 = open('bald_test.txt','w')
i = 1
for row in f_label3:
	# print(row)
	# f_part1.writelines(row)
	if i <= 60000:
		f_part1.writelines(row)
	else:
		f_part2.writelines(row)
	i = i + 1

f_part1.close()
f_part2.close()
f_label3.close()

f_label4 = open('label_mustache.txt','r')
f_part1 = open('mustache_train.txt','w')
f_part2 = open('mustache_test.txt','w')
i = 1
for row in f_label4:
	# print(row)
	# f_part1.writelines(row)
	if i <= 60000:
		f_part1.writelines(row)
	else:
		f_part2.writelines(row)
	i = i + 1

f_part1.close()
f_part2.close()
f_label4.close()