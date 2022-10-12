import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./BinaryWithMomentum.csv', 'r')
df = pd.read_csv('./BinaryWithNoMomentum.csv', 'r')
df = pd.read_csv('./BipolarWithMomentum.csv', 'r')
# df = pd.read_csv('./BipolarWithNoMomentum.csv', 'r')






i = 0
totalepoch = df.shape[0]
epoch = []
loss = []
print(totalepoch)

for i in range (totalepoch):
    loss.append(df.iat[i, 0])
    epoch.append(i)

plt.figure(figsize=(8, 6))

plt.plot(epoch, loss)

plt.xlabel('Epochs')
plt.ylabel('Training Looses')
plt.title('Training Losses VS Epochs')
plt.show()