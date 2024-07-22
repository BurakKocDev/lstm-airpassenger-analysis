import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler

def plot_decompose(decompose_result):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))
    decompose_result.observed.plot(legend=False, ax=ax1, fontsize=20, grid=True, linewidth=3)
    ax1.set_ylabel("Observed", fontsize=20)
    decompose_result.trend.plot(legend=False, ax=ax2, fontsize=20, grid=True, linewidth=3)
    ax2.set_ylabel("Trend", fontsize=20)
    decompose_result.seasonal.plot(legend=False, ax=ax3, fontsize=20, grid=True, linewidth=3)
    ax3.set_ylabel("Seasonal", fontsize=20)
    decompose_result.resid.plot(legend=False, ax=ax4, fontsize=20, grid=True, linewidth=3)
    ax4.set_ylabel("Residual", fontsize=20)
    plt.show()

# Uçuş verilerini yükleme
flight_data = pd.read_csv('AirPassengersne.csv', parse_dates=['Month'], index_col='Month')

# Sütun adlarını yazdırma
print(flight_data.columns)

# Başlangıç verilerini görselleştirme
cm = sns.light_palette('green', as_cmap=True)
flight_data.head(20).style.background_gradient(cmap=cm)

print(flight_data.describe())
print('-'*40)
print(flight_data.head())

# shape ve info fonksiyonlarını doğru kullanma
print(flight_data.shape)
print(flight_data.info())

# Veri görselleştirme
plt.figure(figsize=(12, 5))
plt.title('Month vs Passengers', fontsize=20)
plt.ylabel('Total Passenger', fontsize=20)
plt.xlabel('Months', fontsize=20)
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Plotting the data
plt.plot(flight_data.index, flight_data['#Passengers'], marker='o')
plt.show()

# Decomposition
decomposition = seasonal_decompose(flight_data['#Passengers'], period=12)
plot_decompose(decomposition)

# veri ön işleme
all_data = flight_data['#Passengers'].values.astype(float)
print(all_data)

test_data_size = 12

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]
print(len(train_data))
print(len(test_data))

# Min-Maks ölçeklendirme.
# Bu durumda veri setinin dağılımı doğrusal dönüşüm yoluyla korunur.
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

# eğitim verisi için normalizasyon.
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

# Bu işlev, ham giriş verilerini eğitime uyacak şekilde sıra verilerine dönüştürerek bir demet döndürür.
# İlk 12 ayda seyahat eden yolcu sayısı, 13 aydaki yolcu sayısını öngörüyor.
# Gruptaki ilk değer: 12 aydaki yolcuların sırası (=özellikler)
# Grubun ikinci değeri: 12 aydaki yolcu sayısı olarak tahmin edilen yolcu sayısı (=hedef)
train_window = 12





def create_inout_sequences(input_data, window):
    inout_seq = []
    L = len(input_data)
    for i in range(L-window):
        train_seq = input_data[i:i+window]
        train_label = input_data[i+window:i+window+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


train_inout_seq = create_inout_sequences(train_data_normalized, train_window)


train_inout_seq[:5]

#LSTM ağları, bir zaman serisindeki önemli olaylar arasında bilinmeyen süreli gecikmeler olabileceğinden, 
#zaman serisi verilerine dayalı olarak sınıflandırma, işleme ve tahminlerde bulunma için çok uygundur.
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


#Modelleme basit bir LTSM katmanı kullanılarak yapılır.
#Giriş_boyutu: Giriş dizisi sayısına karşılık gelir. Sıra uzunluğu 12'dir, ancak ayda yalnızca 1 değer vardır, 
#yani toplam yolcu sayısı, dolayısıyla giriş boyutu 1'dir.
#Hidden_layer_size: Gizli katmanların sayısını belirtir.
#Output_size: Çıktıdaki öğe sayısı bir sonraki aydaki yolcu sayısını tahmin ettiğinden çıktı boyutu 1'dir.
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq) ,1, -1))
        predictions = self.linear(lstm_out[:,-1,:])
        return predictions[-1]
    
    
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)    

print(model)



epochs = 500

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
 
        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')



#Test seti son 12 aya ait yolcu verilerini içermektedir. 
#Model, dizi uzunluğu 12'yi kullanarak tahminlerde bulunmak üzere eğitilmiştir. Son 12 aylık verileri tahmin edelim.
fut_pred = 12

test_inputs = train_data_normalized[-train_window:].tolist()
print(test_inputs)


#Test seti 12 iterasyonla çalıştırılır. Yinelemenin sonunda test_inputs listesi 24 giriş içerir. 
#Son 12 madde test seti için tahmin edilen değerlerdir.
model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        test_inputs.append(model(seq).item())
        
        
#Son 12 tahmini kontrol et.
test_inputs[fut_pred:]


#Veri setini eğitim için normalleştirdiğimiz için tahmin edilen değerler de normalleştirildi. 
#Normalleştirilmiş tahmin değerlerini gerçek tahmin değerlerine dönüştürmemiz gerekiyor. 
#Veri kümesini orijinal değerine dönüştürmek için normalleştirmek için kullandığınız min/maks ölçekleyici nesnesinin 
#inverse_transform'unu kullanın.
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
print(actual_predictions)


x = np.arange(132, 144, 1)
print(x)        

#LSTM'nin tahmini turuncu çizgiyle gösterilir. Sonuçlar kesin olmasa da son 12 ayda seyahat eden toplam yolcu 
#sayısındaki dalgalanmalara bakıldığında yükseliş eğilimini tespit etmek mümkün. 
#LSTM katmanında daha fazla sayıda epoch ve daha fazla sayıda nöron kullanılarak daha iyi performans elde edilebilir.
plt.figure(figsize=(12,5))
plt.title('Month vs Passenger',fontsize = 20)
plt.ylabel('Total Passengers',fontsize = 20)
plt.xlabel('Months',fontsize = 20)
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(flight_data['#Passengers'])
plt.plot(x,actual_predictions)



plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)

plt.plot(flight_data['#Passengers'][-train_window:])
plt.plot(x,actual_predictions)
plt.show()

#Orijinal zaman serisinin sahip olduğu zamanlayıcı serisinin trendi, mevsimselliği ve kalıcılığı korunurken 
#tahmin edilen sonuçların öğrenilip öğrenilmediğinin kontrol edilmesi.
flight_data['#Passengers'][:-train_window]
train_df = pd.DataFrame(flight_data['#Passengers'][:-train_window])
actual_df = pd.DataFrame(actual_predictions)
actual_df.columns = ['#Passengers']
new_predict = pd.concat([train_df,actual_df]).reset_index(drop=True)


plt.figure(figsize=(12,5))
plt.title('Month vs Passenger',fontsize = 20)
plt.ylabel('Total Passengers',fontsize = 20)
plt.xlabel('Months',fontsize = 20)
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(new_predict)
plt.plot(flight_data['#Passengers'])

#Tahmin edilen sonuçlarla mevsimsel ayrıştırma analizi yapalım.
decomposition = seasonal_decompose(new_predict, period=12) 
plot_decompose(decomposition)
#Aşağıdaki şekle bakıldığında, basit modele rağmen trendi, mevsimselliği ve kalıcılığı iyi koruyarak tahmin edildiği doğrulanabilir.