import sys
input_data = open(sys.argv[2])
i = -1
while(i<int(sys.argv[1])):
    line = input_data.readline()
    i += 1
line = line.strip(' ')
line = line.rstrip('\n')
line = line.split(' ')
line = [float(i) for i in line]
i = 1
output = ""
for x in sorted(line):
    output = output + str(x)
    if i != len(line):
        output = output + ','
        i += 1
print output


