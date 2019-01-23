import csv
import os

# filename = 'cs1048486361028912.csv'
# with open(filename) as f:
# 	reader = csv.reader(f)
# 	readl = list(reader)
# 	print(readl[12])

root = os.getcwd() + '/cartoonset100k/'

### label of hair color 
# print(os.listdir())

f_label = open('label_haircolor.txt','w')
for i in range(10):
	for name in os.listdir(root + str(i)):
		a,b = os.path.splitext(name)
		# print(addr)
		if b == '.csv':
			addr = root + str(i) + '/' + a + '.png'
			with open(root + str(i) + '/' + name) as f:
				reader = csv.reader(f)
				readlist = list(reader)
				index = readlist[12][1]
				if int(index) == 7:
					index = '0'
				elif int(index)>=1 and int(index)<=4:
					index = '1'
				elif int(index)>=5 and int(index)<=6:
					index = '2'
				else:
					index = '3'
				f.close()
				# f_label = open('label_total.txt','w')
				f_label.write(addr + ' ' + index + '\n')
f_label.close()

### label of glasses
f_label2 = open('label_glasses.txt','w')
for i in range(10):
	for name in os.listdir(root + str(i)):
		a,b = os.path.splitext(name)
		# print(addr)
		if b == '.csv':
			addr = root + str(i) + '/' + a + '.png'
			with open(root + str(i) + '/' + name) as f:
				reader = csv.reader(f)
				readlist = list(reader)
				index = readlist[13][1]
				# print(index)
				if int(index) == 11:
					index = '0'
				else:
					index = '1'
				f.close()
				# f_label = open('label_total.txt','w')
				f_label2.write(addr + ' ' + index + '\n')
f_label2.close()

### label of mustache
f_label3 = open('label_mustache.txt','w')
for i in range(10):
	for name in os.listdir(root + str(i)):
		a,b = os.path.splitext(name)
		# print(addr)
		if b == '.csv':
			addr = root + str(i) + '/' + a + '.png'
			with open(root + str(i) + '/' + name) as f:
				reader = csv.reader(f)
				readlist = list(reader)
				index = readlist[8][1]
				if int(index) == 14:
					index = '0'
				else:
					index = '1'
				f.close()
				# f_label = open('label_total.txt','w')
				f_label3.write(addr + ' ' + index + '\n')
f_label3.close()

### label of bald
f_label4 = open('label_bald.txt','w')
for i in range(10):
	for name in os.listdir(root + str(i)):
	    a,b = os.path.splitext(name)
	    if b == '.csv':
	        addr = root + str(i) + '/' + a + '.png'
	        with open(root + str(i) + '/' + name) as f:
	            reader = csv.reader(f)
	            readlist = list(reader)
	            index = readlist[9][1]
	            if int(index) == 0:
	                index = '0'
	            else:
	                index = '1'
	            f.close()
	            f_label4.write(addr + ' ' + index + '\n')
f_label4.close()
