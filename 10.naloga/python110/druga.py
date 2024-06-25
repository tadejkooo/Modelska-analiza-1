import matplotlib.pyplot as plt
import numpy as np

def desumiraj(x,stevilka):
    frekvencno=np.fft.fft(x)
    f2=abs(frekvencno)**2
    # print(len(frekvencno))
    # plt.plot(x)
    # plt.plot(x1)
    # plt.plot(x2)
    # plt.plot(x3)
    # plt.grid()
    
    # plt.title('izhodni signal')
    # plt.xlabel('čas')
    # plt.savefig('druga_izhodni_signal.png')
    # plt.show()
    # 
    # plt.plot(frekvencno)
    # plt.show()
    # 
    
    meja=50
    vsota=0
    for i in range(meja,len(f2)):
        vsota+=f2[i]
    povprecje=vsota/(len(f2)-meja)
    if stevilka==1:
        povprecje=0
    tau=16
    r=[]
    fi=[]
    for i in range(len(frekvencno)):
        fi.append(f2[i]/(f2[i]+povprecje))
        r.append(np.exp(-i/tau)/(2*tau))
    for i in range(int(len(frekvencno)/2)):
        r[-i]=r[i]
    # print(fi)
    
    R=np.fft.fft(r)
    
    
    invert=np.fft.ifft(frekvencno/R*fi)
    return invert


# def desumiraj(line):
#     x = np.fft.fft(line)
# 
#     moc = abs(x)**2
#     N=len(line)
#     print(N)
# 
#     ##kako močan šum imamo
# 
#     mejna=50
#     vsota=0
#     for i in range(mejna,N):
#         vsota+=moc[i]
#     povprecje = vsota/(N-mejna)
#     #print(povprecje)
# 
#     ##wiener filter+odzivna funkcija/response funkcion
# 
#     fi=[]
#     r=[]
#     tau=30
# 
#     for i in range(N):
#         r.append(np.exp(-i/tau)/tau)
#         fi.append(moc[i]/(moc[i]+povprecje))
#     # for i in range(int(N/2)):
#     #    r[-i]=r[i]
# 
#     R=np.fft.fft(r)
# 
#     # lala=x/R*fi
#     # lala2=np.fft.ifft(lala)
#     # plt.plot(x)
#     # plt.plot(lala)
#     # plt.axhline(y=np.sqrt(povprecje))
#     # plt.legend(['fft-org','fft-filter'])
#     # plt.show()
#     # plt.plot(line)
#     # plt.plot(lala2)
#     # plt.legend(['fft-org','fft-filter'])
#     # plt.show()
#     return np.fft.ifft(x/R*fi)
    
    

x=[]
x1=[]
x2=[]
x3=[]


f = open('data/signal0.dat', 'r',encoding='utf-8')
data_file=f.readlines()


for vrsta in data_file:
    x.append(float(vrsta))


f = open('data/signal1.dat', 'r',encoding='utf-8')
data_file=f.readlines()


for vrsta in data_file:
    x1.append(float(vrsta))


f = open('data/signal2.dat', 'r',encoding='utf-8')
data_file=f.readlines()


for vrsta in data_file:
    x2.append(float(vrsta))



f = open('data/signal3.dat', 'r',encoding='utf-8')
data_file=f.readlines()


for vrsta in data_file:
    x3.append(float(vrsta))




y0=desumiraj(x,1)
y1=desumiraj(x1,0)
y2=desumiraj(x2,0)
y3=desumiraj(x3,0)

# 
# plt.plot(x)
# plt.plot(x1)
# plt.plot(x2)
# # plt.plot(x3)
# plt.grid()
# plt.legend(['signal0','signal1','signal2'])
# plt.xlabel('čas')
# plt.savefig('druga_meritve_delna2.png')
# plt.show()

# plt.plot(y1, 'tab:orange')
# plt.plot(y2, 'tab:green')
# plt.plot(y3, 'tab:red')
plt.plot(y0, 'tab:blue')
plt.xlabel('t')
plt.ylabel(r'$u(t)$')
plt.legend(['signal3','signal1','signal3'])
#plt.savefig('druga_delna_rezultati3.png')
plt.show()

plt.figure()
plt.plot(np.linspace(0,512,512), x3, 'tab:red')
plt.plot(np.linspace(0,512,512), x2, 'tab:green')
plt.plot(np.linspace(0,512,512), x1, 'tab:orange')
plt.plot(np.linspace(0,512,512), x, 'tab:blue')
plt.legend(['signal3', 'signal2', 'signal1', 'signal0'])
plt.xlabel('t')
plt.ylabel(r'$c(t)$')
plt.show()

