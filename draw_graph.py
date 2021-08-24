import matplotlib.pyplot as plt


f = open('evaluation_theoretical_data.txt', 'r')
lines = f.readlines()

loss_train = []
accuracy_train = []
loss_valid = []
accuracy_valid = []

evaluation_list = []
for line in lines:
    evaluation_list.append(line.split())

#order : epoch, train, valid, chkp
for i in range(len(evaluation_list)):
    print(evaluation_list[i])
    if(i%4==1):
        loss = float(evaluation_list[i][-3][:-1])
        accuracy = float(evaluation_list[i][-1])
        loss_train.append(loss)
        accuracy_train.append(accuracy)

    if (i % 4 == 2):
        loss = float(evaluation_list[i][-3][:-1])
        accuracy = float(evaluation_list[i][-1])
        loss_valid.append(loss)
        accuracy_valid.append(accuracy)

plt.plot(loss_train)
plt.plot(loss_valid)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(accuracy_train)
plt.plot(accuracy_valid)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
